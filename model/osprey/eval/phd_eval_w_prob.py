import argparse
import torch
import os
import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from model.osprey.constants import IMAGE_TOKEN_INDEX
from model.osprey.conversation import conv_templates, SeparatorStyle
from model.osprey.mm_utils import tokenizer_image_token
from model.osprey.train.train import preprocess_multimodal
from model.osprey.train.train import DataArguments
from model.osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
import transformers
print(transformers.__version__)
data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True

def annToMask(ann, h, w):
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        print('Using distributed mode: 1')
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
        print('Using distributed mode: slurm')
        print(f"world: {os.environ['WORLD_SIZE']}, rank:{os.environ['RANK']},"
              f" local_rank{os.environ['LOCAL_RANK']}, local_size{os.environ['LOCAL_SIZE']}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class RESDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, tokenizer, img_processor):
        # self.img_folder = img_folder
        self.ann_file = ann_file

        self.label_list = [json.loads(q) for q in open(self.ann_file, 'r')]
        self._ids = range(len(self.label_list))
        
        # self.coco = COCO(annotation_file)
        self.root_path = img_folder
        # self.img_ids = self.coco.getImgIds()
        self.image_processor = img_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label_list)
    @property
    def ids(self):
        return deepcopy(self._ids)
    
    def __getitem__(self, idx):
        label = self.label_list[idx]
        filename = label["image_path"]
        image = Image.open(os.path.join(self.root_path, filename)).convert('RGB')
        question = label["question"]
        height, width = image.size
        round_ids = 0
        caption = {}
        
    
        init_inputs = get_init_inputs(image,
                                    self.image_processor,
                                    question,
                                    round_ids=round_ids,
                                    )
        image = init_inputs['image']

        print('------ALL zero Mask!-------')
        mask = torch.zeros(init_inputs['img_size']).unsqueeze(0).cuda()

        conv = conv_templates['osprey_v1'].copy()
        qs = init_inputs['sources'][0][0]['value']

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return input_ids,image, question,mask, stop_str


def custom_collate_fn(batch):
    input_ids = [item[0] for item in batch]
    image = [item[1] for item in batch]
    question = [item[2] for item in batch]
    mask = [item[3] for item in batch]
    stop_str = [item[4] for item in batch]
    return input_ids, image, question, mask, stop_str
class POPE_EVAL():
    def __init__(self, model_path, args):
        model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.model = OspreyLlamaForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.float32,
                                                ).half().cuda()
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        
        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        # import pdb;pdb.set_trace()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device='cuda')
        dataset = RESDataset(args.img,args.json, self.tokenizer, self.image_processor)
        distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
        self.dataloader = DataLoader(dataset, batch_size=1, num_workers=0,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)
        self.captions_all = []
        # self.gt_all = {}
        # self.gt_all['images'] = []
        # self.gt_all['annotations'] = []
        # import pdb;pdb.set_trace()

    
    def forward(self, gt_file='captions_refcocog_gt.json', caption_file_dir='captions_refcocog_osprey.json', args=None):
        chosed_layer = args.chosed_layer
        
        numbers_list = chosed_layer.split(',')
       
        numbers = [int(num) for num in numbers_list]
        layer_ = numbers

        layer_prob = args.layer_prob
        prob_numbers_list = layer_prob.split(',')
        prob_numbers = [float(num) for num in prob_numbers_list]
        layer_prob = prob_numbers
        layer_prob = torch.tensor(layer_prob).to(dtype=torch.float16).cuda()
        for idx, (input_ids,image, question,mask, stop_str) in enumerate(tqdm(self.dataloader)):
            caption = {}
            
            # print(Using)
            print('-----Candidate Layer {}-----repetition_penalty{}'.format(layer_,str(args.t)))
            with torch.inference_mode():
                self.model.orig_forward = self.model.forward
                self.model.forward = partial(self.model.orig_forward,
                                             img_metas=[None],
                                             masks=[mask[0].half()])
                
                output_ids = self.model.generate(
                    input_ids[0],
                    images=image[0].unsqueeze(0).half().cuda(),
                    max_new_tokens=10,
                    use_cache=True,
                    num_beams=1,# early_exit_layers=1
                    cycd_layers=layer_,
                    repetition_penalty=float(args.t), #[1,1.2]
                    layers_prob=layer_prob
                )

                self.model.forward = self.model.orig_forward

            input_token_len = input_ids[0].shape[1]
            n_diff_input_output = (
                input_ids[0] != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                  skip_special_tokens=True)[0]

            outputs = outputs.strip()
            if outputs.endswith(stop_str[0]):
                outputs = outputs[:-len(stop_str[0])]
            outputs = outputs.strip()
            if ':' in outputs:
                outputs = outputs.split(':')[1]
            
            print(outputs)
            # outputs = outputs.replace('.', '.\n')
            caption['question'] = question
            caption['answer'] = outputs
            # gt['id'] = str(ann['id'])
            # gt['image_id'] = str(ann['id'])
            # gt['caption'] = inputs['caption']
            self.captions_all.append(caption)
        os.makedirs(caption_file_dir, exist_ok=True)
        results_path = f"{caption_file_dir}/{os.path.basename(args.model)}_pope_{args.rank}.json"
        with open(results_path, 'w') as json_file:
            json.dump(self.captions_all, json_file, indent=2)

def get_init_inputs(image,
                    processor,
                    question,
                    round_ids=0,
                    last_round_source=None):

    if round_ids == 0:
        
        h,w = image.size
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

    begin_str = """Taking note of the image <image> and region <mask><pos>.\nPlease give me a straightforward response to '{}' """

    sources = dict()
    sources['conversations'] = []

    sources['conversations'].append({'from': 'human', 'value': begin_str.format(question)})
    
    sources = preprocess_multimodal([sources['conversations']], data_args, cur_token_len)

    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    data_dict['img_size'] = (w,h)
    # data_dict['masks'] = mask
    # print(data_dict)
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liangyaoyuan/pretrained_weight/osprey-7b-regcocog-finetuned')
    # parser.add_argument('--img', help='path to coco imgs', default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liangyaoyuan/github/other_llm_box/glamm_data_seg/GLAMM_data/coco_2014/train2014')
    parser.add_argument('--img', help='path to coco imgs', default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liangyaoyuan/dataset/vilt_dataset/coco/images/val2014')
    parser.add_argument('--t',default='1.05')
    parser.add_argument('--json', help='path to refcocog val json file', default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liangyaoyuan/github/other_llm_box/Chat-UniVi/ChatUniVi/eval/questions/coco_pope/coco_pope_random.jsonl')
    parser.add_argument('--output', help='output json file dir', default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liangyaoyuan/github/other_llm_box/Osprey/osprey/eval/result_dir_refcocog')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--chosed_layer', default='')
    parser.add_argument('--layer_prob', default='')

    args = parser.parse_args()
    init_distributed_mode(args)
    refcocog_eval = POPE_EVAL(args.model, args)
    refcocog_eval.forward( caption_file_dir=args.output, args=args)

