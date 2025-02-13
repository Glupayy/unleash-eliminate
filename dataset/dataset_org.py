import numpy as np
import torch
from typing import Optional, Tuple
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
# from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
# from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
# from dataset.region_datasets.Flickr_Region_ds import Flickr30kRegDataset
# from dataset.segm_datasets.Semantic_Segm_ds import SemanticSegmDataset
# from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset,ReferSegmDataset_CYC
# from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
# from dataset.region_datasets.RefCOCO_VG_Region_ds import (RefCocoRegDataset, RefCocoGRegDataset, RefCocoPRegDataset,
#                                                           VisualGenomeRegDataset)
from utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def custom_collate_fn_ospreyeval_cyc(batch):
    input_ids = [item[0] for item in batch]
    image = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    stop_str = [item[3] for item in batch]
    img_id = [item[4] for item in batch]
    global_enc_image = [item[5] for item in batch]
    grounding_enc_image = [item[6] for item in batch]
    image_resize_res = [item[7] for item in batch]
    org_image_resize_res = [item[8] for item in batch]

    return input_ids, image, masks, stop_str, img_id,global_enc_image,grounding_enc_image,image_resize_res,org_image_resize_res

# def custom_collate_fn_ospreyeval(batch):
#     input_ids = [item[0] for item in batch]
#     image = [item[1] for item in batch]
#     masks = [item[2] for item in batch]
#     stop_str = [item[3] for item in batch]
#     img_id = [item[4] for item in batch]
#     return input_ids, image, masks, stop_str, img_id


# def custom_collate_fn_osprey(batch, tokenizer=None, inference=False, local_rank=-1):
#     # Initializing lists and counters
#     image_path_list = []
#     masks_list = []
#     questions_list, conversation_list = [], []
#     selected_labels_list= []
#     image_list = []
#     inferences=[]
#     # Iterating through the batch
#     for (image_path, masks, sampled_classes, questions, conversations, image_2_reg) in batch:
#         image_path_list.append(image_path)
#         # global_enc_image_list.append(global_enc_image)
#         # grounding_enc_image_list.append(grounding_enc_image)
#         # bboxes_list.append(bboxes)
#         conversation_list.extend(conversations)
#         masks_list.append([] if masks is None else masks.float())
#         # label_list.append(label)
#         # resize_list.append(resize)
#         questions_list.append(questions)
#         selected_labels_list.append(sampled_classes)
#         image_list.append(image_2_reg)
#         # offset_list.append(cnt := cnt + len(conversations))
#         inferences.append(inference)
#         # dataset_class.append(data_class)
#     print(conversation_list)
#     # Handling the conversation list
#     # if use_mm_start_end:
#     #     replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#     #     conversation_list = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list]

#     # Tokenizing and padding input ids
#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list],
#         batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     attention_masks = input_ids.ne(tokenizer.pad_token_id)

#     # Preparing targets and handling conversation types
#     conv = conversation_lib.default_conversation.copy()
#     # print(conv)
#     targets = input_ids.clone()
#     # conv_type == "llava_v1"
#     sep = conv.sep + conv.roles[1] + ": "
#     sep2 = conv.sep2
#     import pdb; pdb.set_trace()
#     for conversation, target in zip(conversation_list, targets):
#         _process_conversation(conversation, target, tokenizer, sep, sep2)

#     # Adjusting for inferences
#     if not inferences[0]:
#         truncate_len = tokenizer.model_max_length - 575
#         if input_ids.shape[1] > truncate_len:
#             input_ids, targets, attention_masks = map(
#                 lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
#                 )
#     # print(dataset_class)
#     # if 'RefCOCO' in dataset_class or 'RefCOCOP' in dataset_class:
#     #     import pdb;pdb.set_trace()
#     return {"image_paths": image_path_list,"input_ids": input_ids, "labels": targets,
#         "attention_masks": attention_masks, "masks_list": masks_list, "questions_list": questions_list,
#         "sampled_classes_list": selected_labels_list, "inference": inferences[0],
#         "conversation_list": conversation_list, 'img_reg':image_list}

# def custom_collate_fn_cyc(batch, tokenizer=None, use_mm_start_end=True, inference=False, decode_sent_train=False, local_rank=-1,tokenizer_reg=None):
#     # Initializing lists and counters
#     # import pdb;pdb.set_trace() # 在传进来之前tokenizer res就烂掉了
#     image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
#     bboxes_list, conversation_list, masks_list = [], [], []
#     label_list, resize_list, questions_list = [], [], []
#     selected_labels_list, offset_list, inferences = [], [0], []
#     cnt = 0
#     conversation_list_reg = []
#     questions_list_reg = []
#     label_reg_list = []
#     bboxes_org_list = []
#     decode_list = []
#     img_reg_list = []
#     # dataset_class = []
#     # Iterating through the batch
#     for (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, resize, questions,
#          selected_labels, questions_reg,conversations_reg,label_reg,bboxes_org,img_reg) in batch:
        
#         image_path_list.append(image_path)
#         global_enc_image_list.append(global_enc_image)
#         grounding_enc_image_list.append(grounding_enc_image)
#         bboxes_list.append(bboxes)
#         bboxes_org_list.append(bboxes_org)

#         conversation_list.extend(conversations)
#         conversation_list_reg.extend(conversations_reg)

#         masks_list.append([] if masks is None else masks.float())
#         label_list.append(label)
#         label_reg_list.append(label)
#         resize_list.append(resize)
#         questions_list.append(questions)
#         questions_list_reg.append(questions_reg)
#         selected_labels_list.append(selected_labels)
#         offset_list.append(cnt := cnt + len(conversations))
#         inferences.append(inference)
#         decode_list.append(decode_sent_train)
#         img_reg_list.append(img_reg)
#         # dataset_class.append(data_class)
#     # import pdb;pdb.set_trace()
#     # Handling the conversation list
#     if use_mm_start_end:
#         replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#         conversation_list = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list]
#     # osprey don't need to replace <image>
#     # conversation_list_reg = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list_reg]
#     # import pdb;pdb.set_trace()
#     # Tokenizing and padding input ids
#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list],
#         batch_first=True, padding_value=tokenizer.pad_token_id
#     )

#     attention_masks = input_ids.ne(tokenizer.pad_token_id)

#     input_ids_reg = torch.nn.utils.rnn.pad_sequence(
#         [tokenizer_image_token(prompt, tokenizer_reg, return_tensors="pt") for prompt in conversation_list_reg],
#         batch_first=True, padding_value=tokenizer_reg.pad_token_id
#     )

#     attention_masks_reg = input_ids_reg.ne(tokenizer_reg.pad_token_id)

#     # Preparing targets and handling conversation types
#     conv = conversation_lib.default_conversation.copy()
    
#     # print(conv)
#     targets = input_ids.clone()
#     targets_reg = input_ids_reg.clone()
#     # conv_type == "llava_v1"
#     sep = conv.sep + conv.roles[1] + ": "
#     sep2 = conv.sep2
#     # import pdb;pdb.set_trace()
#     for conversation, target in zip(conversation_list, targets):
#         _process_conversation(conversation, target, tokenizer, sep, sep2)
#     # import pdb;pdb.set_trace()
    
#     for conversation, target in zip(conversation_list_reg, targets_reg):
#         _process_conversation(conversation, target, tokenizer_reg, sep, sep2)
#     # Adjusting for inferences
#     if not inferences[0]:
#         truncate_len = tokenizer.model_max_length - 575
#         if input_ids.shape[1] > truncate_len:
#             input_ids, targets, attention_masks = map(
#                 lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
#                 )
#         if input_ids_reg.shape[1] > truncate_len:
#             input_ids_reg, targets_reg, attention_masks_reg = map(
#                 lambda x: x[:, :truncate_len], [input_ids_reg, targets_reg, attention_masks_reg]
#                 )
#     # print(dataset_class)
#     # if 'RefCOCO' in dataset_class or 'RefCOCOP' in dataset_class:
#     # import pdb;pdb.set_trace()
#     return {"image_paths": image_path_list, "global_enc_images": torch.stack(global_enc_image_list, dim=0), #2,3,336,336
#         "grounding_enc_images": torch.stack(grounding_enc_image_list, dim=0), # 2,3,1024,1024
#         "bboxes": None if bboxes_list[0] is None else bboxes_list, "bboxes_org": None if bboxes_org_list[0] is None else bboxes_org_list, "input_ids": input_ids, "labels": targets,
#         "attention_masks": attention_masks, "masks_list": masks_list, "label_list": label_list,
#         "resize_list": resize_list, "offset": torch.LongTensor(offset_list), "questions_list": questions_list,
#         "sampled_classes_list": selected_labels_list, "inference": inferences[0],
#         "conversation_list": conversation_list, 
#         "conversation_list_reg": conversation_list_reg, "input_ids_reg": input_ids_reg, "labels_reg": targets_reg,'label_list_reg':label_reg_list,
#         "questions_list_reg": questions_list_reg, "attention_masks_reg": attention_masks_reg, "decode_sent_train": decode_list[0],'img_reg': img_reg_list}

# def _process_conversation(conversation, target, tokenizer, sep, sep2):
#     # import pdb;pdb.set_trace()
#     total_len = target.ne(tokenizer.pad_token_id).sum().item()
#     rounds = conversation.split(sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_INDEX

#     for rou in rounds:
#         if not rou:
#             break
#         # print()
#         parts = rou.split(sep)
#         # print(len(parts), rou, conversation, target,sep,sep2,rounds)
#         assert len(parts) == 2, (len(parts), rou, conversation, target,sep,sep2,rounds)
#         parts[0] += sep

#         if DEFAULT_IMAGE_TOKEN in conversation:
#             round_len = len(tokenizer_image_token(rou, tokenizer))
#             instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#         else:
#             round_len = len(tokenizer(rou).input_ids)
#             instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#         target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
#         cur_len += round_len

#     target[cur_len:] = IGNORE_INDEX
#     if cur_len < tokenizer.model_max_length:
#         assert cur_len == total_len
