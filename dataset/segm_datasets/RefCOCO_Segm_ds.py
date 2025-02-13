import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.osprey.conversation import conv_templates, SeparatorStyle
from model.osprey.constants import IMAGE_TOKEN_INDEX
from model.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.grefer import G_REFER
from dataset.utils.refcoco_refer import REFER
from utils.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS, REGION_QUESTIONS
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from model.llava.mm_utils import tokenizer_image_token
def annToMask(ann, h, w):
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m
class ReferSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 3,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',
                 random_sampling=True, inference=False, fix_prompt=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST if not fix_prompt else [ANSWER_LIST[fix_prompt-1]]
        print(self.answer_list)
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation
        self.split = split
        self.initialize_refer_segm_data(refer_segm_data, inference)
        self.random_sampling = random_sampling

    def initialize_refer_segm_data(self, refer_segm_data, inference=False):

        dataset_dir = os.path.join(self.dataset_dir, "Refer_Segm")
        self.refer_seg_ds_list = refer_segm_data.split("||")
        # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_segm_data = {}

        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = G_REFER(dataset_dir, dataset_name, splitBy) if dataset_name == "grefcoco" else\
                REFER(dataset_dir, dataset_name, splitBy)
            ref_ids_train = refer_api.getRefIds(split=self.split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            refer_seg_ds = {
                "images": self.load_images(refer_api, images_ids_train, dataset_dir, dataset_name, inference=inference),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_train)
            }

            print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                  f"images and {len(refer_seg_ds['annotations'])} annotations.")
            print(f'\033[92m----SEG-{"Val" if self.validation else "Train"}:'
                  f' Loaded ReferSeg - {dataset_name} dataset ----\033[0m')

            self.refer_segm_data[dataset_name] = refer_seg_ds

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        # Limiting images to 1000(optional) for validation
        loaded_images = loaded_images[:1000] if (self.validation and not inference) else loaded_images
        for item in loaded_images:
            item = item.copy()
            if dataset_name == 'refclef':
                item["file_name"] = os.path.join(dataset_dir, "images", "saiapr_tc-12", item["file_name"])
            else:
                item["file_name"] = os.path.join(dataset_dir.replace("Refer_Segm/", ""), "coco_2014/train2014",
                                                 item["file_name"])
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def __len__(self):
        return self.epoch_samples

    def _set_len(self, length):
        self.epoch_samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        questions = []
        answers = []
        for i, label in enumerate(labels):
            label = label.strip()
            assert len(label.split("||")) == 1
            question_template = random.choice(self.question_templates)
            questions.append(question_template.format(class_name=label.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        dataset_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
        dataset_name = self.refer_seg_ds_list[dataset_idx]
        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        selected_labels = sampled_sents

        # Load and process the image
        image_path = image_info["file_name"]
        # print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # Generate questions and answers
        questions, conversations = self.create_conversations(selected_labels)

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        # set bboxes to None for segmentation datasets
        bboxes = None
        # import pdb;pdb.set_trace()
        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations, masks, label,
                image_resize, questions,selected_labels)

# if __name__ =="__main__":
    
DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

class RESDataset_Osprey(torch.utils.data.Dataset): #for eval
    def __init__(self, annotation_file, root_path, tokenizer):
        self.coco = COCO(annotation_file)
        self.root_path = root_path
        self.img_ids = self.coco.getImgIds()
        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        self.tokenizer = tokenizer
        # self.data_args = DataArguments()
        self.mm_use_im_start_end = False
        self.is_multimodal = True
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_data = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root_path, img_data['file_name'])
        # image = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        annotations_ids = self.coco.getAnnIds([img_data['id']])
        annotations = self.coco.loadAnns(annotations_ids)
        height = img_data['height']
        width = img_data['width']
        round_ids = 0
        last_source = dict()
        caption = {}
        gt = {}
        ann = annotations[0]
        mask_r = ann['segmentation']

        if isinstance(mask_r, list):
            mask = annToMask(mask_r, height, width)
        else:
            mask = maskUtils.decode(mask_r)
        mask = torch.from_numpy(mask).unsqueeze(0)
    
        init_inputs = self.get_init_inputs(img_path,
                                    self.image_processor,
                                    self.tokenizer,
                                    mask=mask,
                                    round_ids=round_ids,
                                    last_round_source=last_source,
                                    )
        image = init_inputs['image']

        masks = init_inputs['masks'].cuda()

        conv = conv_templates['osprey_v1'].copy()
        qs = init_inputs['sources'][0][0]['value']

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # annotations_ids = self.coco.getAnnIds([inputs['id']])
        # annotations = self.coco.loadAnns(annotations_ids)
        # data = self.coco.loadImgs([img])[0]
        # ...处理annotations...
        return input_ids, image, masks, stop_str, str(ann['id'])

    def get_init_inputs(self, img_path,
                    processor,
                    pred_bboxes,
                    mask,
                    round_ids=0,
                    last_round_source=None):

        if round_ids == 0:
            # import pdb;pdb.set_trace()
            image = Image.open(img_path).convert('RGB')

            image = processor.preprocess(image,
                                        do_center_crop=False,
                                        return_tensors='pt')['pixel_values'][0] # (3,512,767)

            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                    size=(512, 512),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0) # 3,512,512
            
        else:
            image = last_round_source['image']

        cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

        mask = mask.to(image.device)

        begin_str = """<image>.\nThis provides an overview of the picture.\n"""

        sources = dict()
        sources['conversations'] = []
        question = 'Can you give me a description of the region <mask><pos>?'

        sources['conversations'].append({'from': 'human', 'value': begin_str+question})
        
        sources = self.preprocess_multimodal([sources['conversations']], self.mm_use_im_start_end , cur_token_len)

        data_dict = {}
        data_dict['sources'] = sources
        data_dict['image'] = image
        data_dict['masks'] = mask
        # import pdb;pdb.set_trace()
        # print(data_dict)
        return data_dict
    def preprocess_multimodal(self, 
            sources,
            mm_use_im_start_end,
            cur_token_len
        ):

        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return sources


class RESDataset_Osprey_RES_REG(torch.utils.data.Dataset): #for eval
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    
    def __init__(self, annotation_file, root_path, tokenizer, image_size, image_encoder_path):
        self.coco = COCO(annotation_file)
        self.root_path = root_path
        self.img_ids = self.coco.getImgIds()
        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        self.tokenizer = tokenizer
        # self.data_args = DataArguments()
        self.mm_use_im_start_end = False
        self.is_multimodal = True
        global_image_encoder = image_encoder_path

        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.transform = ResizeLongestSide(image_size)
    
    def __len__(self):
        return len(self.img_ids)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        IMG_SIZE = 1024
        x = (x - IMG_MEAN) / IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
        return x

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_data = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root_path, img_data['file_name'])
        # image_path = data_item['image_path']
        
        # process res inputs:
        image_res = cv2.imread(img_path)
        image_res = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)

        org_image_resize_res = image_res.shape[:2]
        
        # original_size_list = [image_np.shape[:2]]
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image_res, return_tensors="pt")["pixel_values"][0]
        image_res = self.transform.apply_image(image_res)
        image_resize_res = image_res.shape[:2]
        # Prepare input for Grounding Image Encoder
        # import pdb;pdb.set_trace()
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image_res).permute(2, 0, 1).contiguous())
        # import pdb;pdb.set_trace()

        # image = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        annotations_ids = self.coco.getAnnIds([img_data['id']])
        annotations = self.coco.loadAnns(annotations_ids)
        height = img_data['height']
        width = img_data['width']
        round_ids = 0
        last_source = dict()
        caption = {}
        gt = {}
        ann = annotations[0]
        mask_r = ann['segmentation']

        if isinstance(mask_r, list):
            mask = annToMask(mask_r, height, width)
        else:
            mask = maskUtils.decode(mask_r)
        mask = torch.from_numpy(mask).unsqueeze(0)
    
        init_inputs = self.get_init_inputs(img_path,
                                    self.image_processor,
                                    self.tokenizer,
                                    mask=mask,
                                    round_ids=round_ids,
                                    last_round_source=last_source,
                                    )
        image = init_inputs['image']

        masks = init_inputs['masks'].cuda()

        conv = conv_templates['osprey_v1'].copy()
        qs = init_inputs['sources'][0][0]['value']

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # annotations_ids = self.coco.getAnnIds([inputs['id']])
        # annotations = self.coco.loadAnns(annotations_ids)
        # data = self.coco.loadImgs([img])[0]
        # ...处理annotations...

        # self.get_res_prompt()
        return input_ids, image, masks, stop_str, str(ann['id']), global_enc_image,grounding_enc_image,image_resize_res,org_image_resize_res

    # def get_res_prompt()
    # #nouse
    #     input_str = input_str.replace('&lt;', '<').replace('&gt;', '>')
    #     prompt = input_str
    #     prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + prompt
    #     if args.use_mm_start_end:
    #         replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    #         prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    #     if not follow_up:
    #         conv.append_message(conv.roles[0], prompt)
    #         conv.append_message(conv.roles[1], "")
    
    #     prompt = conv.get_prompt()
    #     return 0
    def get_init_inputs(self, img_path,
                    processor,
                    pred_bboxes,
                    mask,
                    round_ids=0,
                    last_round_source=None):

        if round_ids == 0:
            # import pdb;pdb.set_trace()
            image = Image.open(img_path).convert('RGB')

            image = processor.preprocess(image,
                                        do_center_crop=False,
                                        return_tensors='pt')['pixel_values'][0] # (3,512,767)

            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                    size=(512, 512),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0) # 3,512,512
            
        else:
            image = last_round_source['image']

        cur_token_len = (image.shape[1] // 16) * (image.shape[2] // 16)

        mask = mask.to(image.device)

        begin_str = """<image>.\nThis provides an overview of the picture.\n"""

        sources = dict()
        sources['conversations'] = []
        question = 'Can you give me a description of the region <mask><pos>?'

        sources['conversations'].append({'from': 'human', 'value': begin_str+question})
        
        sources = self.preprocess_multimodal([sources['conversations']], self.mm_use_im_start_end , cur_token_len)

        data_dict = {}
        data_dict['sources'] = sources
        data_dict['image'] = image
        data_dict['masks'] = mask
        # import pdb;pdb.set_trace()
        # print(data_dict)
        return data_dict
    def preprocess_multimodal(self, 
            sources,
            mm_use_im_start_end,
            cur_token_len
        ):

        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return sources


class ReferSegmDataset_Osprey_REG(torch.utils.data.Dataset):
    
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 1,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',
                 random_sampling=True, inference=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        # self.question_templates_REG = DETAILED_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        # self.begin_str_reg = """<image>\nThis provides an overview of the picture.\n"""
        self.begin_str_reg = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'
        self.validation = validation
        self.split = split
        self.initialize_refer_segm_data(refer_segm_data, inference)
        self.random_sampling = random_sampling
        self.image_processor_reg = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        # self.intro_string_reg = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        

    def initialize_refer_segm_data(self, refer_segm_data, inference=False):

        dataset_dir = os.path.join(self.dataset_dir, "Refer_Segm")
        self.refer_seg_ds_list = refer_segm_data.split("||")
        # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_segm_data = {}

        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = G_REFER(dataset_dir, dataset_name, splitBy) if dataset_name == "grefcoco" else\
                REFER(dataset_dir, dataset_name, splitBy)
            ref_ids_train = refer_api.getRefIds(split=self.split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            refer_seg_ds = {
                "images": self.load_images(refer_api, images_ids_train, dataset_dir, dataset_name, inference=inference),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_train)
            }

            print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                  f"images and {len(refer_seg_ds['annotations'])} annotations.")
            print(f'\033[92m----SEG-{"Val" if self.validation else "Train"}:'
                  f' Loaded ReferSeg - {dataset_name} dataset ----\033[0m')

            self.refer_segm_data[dataset_name] = refer_seg_ds

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        # Limiting images to 1000(optional) for validation
        # loaded_images = loaded_images[:1000] if (self.validation and not inference) else loaded_images
        for item in loaded_images:
            item = item.copy()
            if dataset_name == 'refclef':
                item["file_name"] = os.path.join(dataset_dir, "images", "saiapr_tc-12", item["file_name"])
            else:
                item["file_name"] = os.path.join(dataset_dir.replace("Refer_Segm/", ""), "coco_2014/train2014",
                                                 item["file_name"])
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def __len__(self):
        return self.epoch_samples

    def _set_len(self, length):
        self.epoch_samples = length


    def create_conversations_reg(self, labels): # question 是template，label就是原始dataset 的caption，没有加别的处理
        questions = []
        answers = []
        for i, label in enumerate(labels):
            question = '<region>'
            question = question.replace('<region>', '<mask><pos>')
            questions.append(question)
            answers.append(label)
        # import pdb;pdb.set_trace()
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str_reg + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        # print(conversations)
        return questions, conversations

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        dataset_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
        dataset_name = self.refer_seg_ds_list[dataset_idx]
        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample: # 句子要是超过3条就只取3条的mask, 每次只取1条，我只需要拿到gt_bbox, gt_caption,
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        selected_labels = sampled_sents
        # if self.split=='val':
        #     print(selected_labels)
        # Load and process the image
        image_path = image_info["file_name"]
        # print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = Image.fromarray(image)
        
        image_2_reg = self.image_processor_reg.preprocess(img_rgb,
                                        do_center_crop=False,
                                        return_tensors='pt')['pixel_values'][0]

        image_2_reg = torch.nn.functional.interpolate(image_2_reg.unsqueeze(0),
                                                    size=(512, 512),
                                                    mode='bilinear',
                                                    align_corners=False) 
        # print(image_2_reg.size())
        # orig_h, orig_w = image.shape[:2]
        # global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        # image = self.transform.apply_image(image)
        # image_resize = image.shape[:2]
        # grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # post_h, post_w = global_enc_img.shape[1:3]
        # Generate questions and answers
        # questions, conversations = self.create_conversations(selected_labels)
        questions_reg, conversations_reg = self.create_conversations_reg(selected_labels)
        flag = False
        masks = []
        # bboxes_list=[]
        # bboxes_norm=None
        # bbox = None
        assert len(sampled_ann_ids) == 1, len(sampled_ann_ids)
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
            # bbox = self._get_valid_bbox(ann['bbox'], image_info["width"], image_info["height"])
            # orig_h, orig_w = image_resize
            # bbox_norm = self.region_enc_processor((orig_h, orig_w), (post_h, post_w), bbox, 
                                                            # global_enc_img.device)
            
            # bboxes_list.append(bbox_norm)

        masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        # label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        # label_reg = torch.ones(grounding_enc_img.shape[1], grounding_enc_img.shape[2]) * self.IGNORE_LABEL

        # print(selected_labels)
        # print(image_path)
        return (image_path, masks, selected_labels, questions_reg, conversations_reg, image_2_reg) # selected_labels


class ReferSegmDataset_CYC(torch.utils.data.Dataset):
    
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 1,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',
                 random_sampling=True, inference=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.question_templates_REG = DETAILED_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.begin_str_reg = '<image>\nI will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'
        self.validation = validation
        self.split = split
        self.initialize_refer_segm_data(refer_segm_data, inference)
        self.random_sampling = random_sampling
        self.image_processor_reg = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        # self.intro_string_reg = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        

    def initialize_refer_segm_data(self, refer_segm_data, inference=False):

        dataset_dir = os.path.join(self.dataset_dir, "Refer_Segm")
        self.refer_seg_ds_list = refer_segm_data.split("||")
        # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_segm_data = {}

        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = G_REFER(dataset_dir, dataset_name, splitBy) if dataset_name == "grefcoco" else\
                REFER(dataset_dir, dataset_name, splitBy)
            ref_ids_train = refer_api.getRefIds(split=self.split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            refer_seg_ds = {
                "images": self.load_images(refer_api, images_ids_train, dataset_dir, dataset_name, inference=inference),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_train)
            }

            print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                  f"images and {len(refer_seg_ds['annotations'])} annotations.")
            print(f'\033[92m----SEG-{"Val" if self.validation else "Train"}:'
                  f' Loaded ReferSeg - {dataset_name} dataset ----\033[0m')

            self.refer_segm_data[dataset_name] = refer_seg_ds

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        # Limiting images to 1000(optional) for validation
        # loaded_images = loaded_images[:1000] if (self.validation and not inference) else loaded_images
        for item in loaded_images:
            item = item.copy()
            if dataset_name == 'refclef':
                item["file_name"] = os.path.join(dataset_dir, "images", "saiapr_tc-12", item["file_name"])
            else:
                item["file_name"] = os.path.join(dataset_dir.replace("Refer_Segm/", ""), "coco_2014/train2014",
                                                 item["file_name"])
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def __len__(self):
        return self.epoch_samples

    def _set_len(self, length):
        self.epoch_samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        questions = []
        answers = []
        # import pdb;pdb.set_trace()
        for i, label in enumerate(labels):
            label = label.strip()
            assert len(label.split("||")) == 1
            question_template = random.choice(self.question_templates)
            questions.append(question_template.format(class_name=label.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations

    def create_conversations_reg(self, labels, question_templates): # question 是template，label就是原始dataset 的caption，没有加别的处理
        questions = []
        answers = []
        for i, label in enumerate(labels):
            question = '<region>'
            question = question.replace('<region>', '<mask><pos>')
            questions.append(question)
            answers.append(label)
        # import pdb;pdb.set_trace()
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str_reg + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        # print(conversations)
        return questions, conversations

    def _get_valid_bbox(self, bbox, img_width, img_height):
        # import pdb;pdb.set_trace()
        x1, y1, w, h = bbox
        inter_w = max(0, min(x1 + w, img_width) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_height) - max(y1, 0))
        if inter_w * inter_h == 0:
            return None
        return [x1, y1, x1 + w, y1 + h] #xyxy

    def region_enc_processor(self, orig_size, post_size, bboxes,  device):
        orig_h, orig_w = orig_size
        post_h, post_w = post_size
        y_scale = post_h / orig_h
        x_scale = post_w / orig_w
        # shuffle_ids = torch.randperm(len(labels))[:self.max_gt_per_img]
        # selected_bboxes = bboxes[]
        # import pdb;pdb.set_trace()
        # Ensure selected_bboxes is two-dimensional
        # if len(bboxes.shape) == 1:
        #     bboxes = np.expand_dims(bboxes, axis=0)
        bboxes = np.array(bboxes)
        bboxes = np.expand_dims(bboxes, axis=0)
        # selected_labels = [labels[i] for i in shuffle_ids]
        bboxes[:, [0, 2]] *= x_scale
        bboxes[:, [1, 3]] *= y_scale
        bboxes = torch.tensor(bboxes, device=device, dtype=torch.float32) / post_h
        # import pdb;pdb.set_trace()
        return bboxes

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        dataset_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
        dataset_name = self.refer_seg_ds_list[dataset_idx]
        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample: # 句子要是超过3条就只取3条的mask, 每次只取1条，我只需要拿到gt_bbox, gt_caption,
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        selected_labels = sampled_sents

        # Load and process the image
        image_path = image_info["file_name"]
        # print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb = Image.fromarray(image)
        
        image_2_reg = self.image_processor_reg.preprocess(img_rgb,
                                        do_center_crop=False,
                                        return_tensors='pt')['pixel_values'][0]

        image_2_reg = torch.nn.functional.interpolate(image_2_reg.unsqueeze(0),
                                                    size=(512, 512),
                                                    mode='bilinear',
                                                    align_corners=False) 
        # print(image_2_reg.size())
        orig_h, orig_w = image.shape[:2]
        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        post_h, post_w = global_enc_img.shape[1:3]
        # Generate questions and answers
        questions, conversations = self.create_conversations(selected_labels)
        questions_reg, conversations_reg = self.create_conversations_reg(selected_labels, self.question_templates_REG)
        flag = False
        masks = []
        # bboxes_list=[]
        bboxes_norm=None
        bbox = None
        assert len(sampled_ann_ids) == 1, len(sampled_ann_ids)
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
            bbox = self._get_valid_bbox(ann['bbox'], image_info["width"], image_info["height"])
            # orig_h, orig_w = image_resize
            bbox_norm = self.region_enc_processor((orig_h, orig_w), (post_h, post_w), bbox, 
                                                            global_enc_img.device)
            
            # bboxes_list.append(bbox_norm)

        masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        label_reg = torch.ones(grounding_enc_img.shape[1], grounding_enc_img.shape[2]) * self.IGNORE_LABEL

        # print(selected_labels)
        # print(image_path)
        return (image_path, global_enc_img, grounding_enc_img, bbox_norm, conversations, masks, label,
                image_resize, questions,selected_labels, questions_reg,conversations_reg,label_reg, bbox, image_2_reg) # selected_labels


