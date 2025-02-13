import torch
import re,os
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import random
from functools import partial
# from dataset.dataset_org import _process_conversation
from model.osprey.mm_utils import tokenizer_image_token
from model.llava import conversation as conversation_lib
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS, REGION_QUESTIONS
from utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from transformers import AutoTokenizer, CLIPImageProcessor
from model.osprey.model import *
from typing import Optional, Tuple
# import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class GLaMMBaseModel:
    def __init__(self, config, **kwargs):
        super(GLaMMBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_glamm_model(self.config)

    def initialize_glamm_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self.grounding_encoder.mask_decoder.bfloat16()
        self._configure_grounding_encoder(config)
        # self.grounding_encoder.to
        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class GLaMMModel(GLaMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GLaMMModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False

class GLaMMForCausalLM_RES(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tokenizer = None
        self.vis_token=False

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0)) #sam vit 

    def forward(self, new_input=None,**kwargs):
        if new_input:
            return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward_eval_lang(new_input,**kwargs)
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward_eval_lang(self, new_input: dict ,global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, **kwargs, ):
        
        torch.cuda.empty_cache()
        image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
        assert image_embeddings.shape[0] == len(new_input['offset']) - 1
       
        # Create segmentation token mask
        seg_token_mask = self._create_seg_token_mask(new_input['input_ids'])

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(new_input['input_ids'], global_enc_images, new_input['attention_masks'])
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, None, new_input['input_ids'], new_input['labels'], new_input['attention_masks'], new_input['offset'] # bbox = None
            )

        # Process hidden states
        hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, new_input['offset'])
        # import pdb;pdb.set_trace()
        # Generate and post-process masks
        pred_masks = self._generate_and_postprocess_masks(
            pred_embeddings, image_embeddings, resize_list, label_list
        )
        predicted_masks=[]
        for i in range(len(pred_masks)): 
            predicted_masks.append((pred_masks[i] > 0).int())
       
        if inference:
            return {"pred_masks": pred_masks, "gt_masks": masks_list, 'processed_masks':predicted_masks}
        return {"pred_masks": pred_masks, "gt_masks": masks_list, 'processed_masks':predicted_masks,'loss_dict':self._calculate_losses(pred_masks, masks_list, output)}
    
    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, **kwargs, ):
        # Extract grounding encoder image embeddings
        # import pdb;pdb.set_trace()
        image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
        assert image_embeddings.shape[0] == len(offset) - 1
        
        # Create segmentation token mask
        seg_token_mask = self._create_seg_token_mask(input_ids)
        
        seg_list_layers = []
        
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, None, input_ids, labels, attention_masks, offset # bbox = None
            )
            if self.vis_token:
                for idx_h in range(len(output_hidden_states)):
                    seg_i = output_hidden_states[idx_h][seg_token_mask].to(dtype=torch.float32).detach().cpu().numpy()
                    seg_list_layers.append(seg_i)
        # Process hidden states
        hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, offset)
        # import pdb;pdb.set_trace()
        # Generate and post-process masks
        pred_masks = self._generate_and_postprocess_masks(
            pred_embeddings, image_embeddings, resize_list, label_list
        )

        predicted_masks=[]
        for i in range(len(pred_masks)): 
            predicted_masks.append((pred_masks[i] > 0).int())
        # bboxes_pred,bbox_pred_org = self.calculate_bboxes_from_masks_gpu(predicted_masks)
        
        if self.vis_token:
            return {"pred_masks": pred_masks, "gt_masks": masks_list, 'processed_masks':predicted_masks, 'seg_token_feat':seg_list_layers}

        if inference:
            return {"pred_masks": pred_masks, "gt_masks": masks_list, 'processed_masks':predicted_masks}
        return {"pred_masks": pred_masks, "gt_masks": masks_list, 'processed_masks':predicted_masks,'loss_dict':self._calculate_losses(pred_masks, masks_list, output)}
       
        # import pdb;pdb.set_trace()
        # Calculate losses
        # return 

    def _create_seg_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.seg_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, bboxes, input_ids, labels, attention_masks, offset):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_list = bboxes

        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, bboxes=bboxes_list, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        # output_hidden_states: [2,713,4096], seg_token_mask: [2,713], 
        # import pdb;pdb.set_trace()
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])] # self.model.text_hidden_fcs: 4096,256
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask] # 6,256
        seg_token_counts = seg_token_mask.int().sum(-1)
        
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset] # [0,3,6]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            ) # [3,1,256]; [3, 256, 64, 64]
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            # dec_dict = self.model.grounding_encoder.mask_decoder.state_dict()
            # key_list = list(dec_dict.keys())
            # nan_list = [dec_dict[_onekey].isnan().any() for _onekey in key_list]
            # nan_t=torch.tensor(nan_list)
            # import pdb;pdb.set_trace()
            
            self.model.grounding_encoder.mask_decoder.bfloat16()
            low_res_masks, mask_score = self.model.grounding_encoder.mask_decoder( #3 1 256 256
                image_embeddings=image_embeddings[i].unsqueeze(0).bfloat16(),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe().bfloat16(),
                sparse_prompt_embeddings=sparse_embeddings.bfloat16(), dense_prompt_embeddings=dense_embeddings.bfloat16(),
                multimask_output=False, )
            # import pdb;pdb.set_trace()
            
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            low_res_masks = low_res_masks.half()
            # import pdb;pdb.set_trace()
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, ) # [3, 1, 427, 640]
            pred_masks.append(pred_mask[:, 0])
        return pred_masks, mask_score

    def _calculate_losses(self, pred_masks, masks_list, output):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        # Initialize loss components
        # print("res ce loss org: {}".format(output.loss))
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        # Iterate over batch and compute mask-related losses
        for batch_idx, pred_mask in enumerate(pred_masks):
            if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                gt_mask = masks_list[batch_idx]
                # Resize gt_mask to match pred_mask if needed
                if gt_mask.shape[0] != pred_mask.shape[0]:
                    gt_mask = gt_mask[:pred_mask.shape[0]]

                assert gt_mask.shape[0] == pred_mask.shape[
                    0], f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                # Compute Binary Cross-Entropy Loss
                mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) *
                                  gt_mask.shape[0])
                # Compute Dice Loss
                mask_dice_loss += (
                        calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss, }

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
                 bboxes=None, ):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True)
            # import pdb;pdb.set_trace()
            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences
            # import pdb;pdb.set_trace()
            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 575), dtype=torch.bool).cuda(), seg_token_mask], dim=1, )
            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            # import pdb;pdb.set_trace()
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks, mask_score = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks, mask_score


class GLaMMForCausalLMCycle_RES_REG(nn.Module):
    def __init__(self, path_res, path_reg, model_args_res, tokenizer_res, tokenizer_reg, loss_reg_weight):
        # self._set_model_configurations(config, kwargs)
        super().__init__()
        self.model_res = GLaMMForCausalLM_RES.from_pretrained(
            path_res, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args_res
        ).half()
        
        self.model_reg = OspreyLlamaForCausalLM.from_pretrained(
            path_reg,torch_dtype=torch.bfloat16,
        ).half()
        
        self.tokenizer_res = tokenizer_res
        self.tokenizer_reg = tokenizer_reg
        
        for m in self.model_reg.modules():
            m.tokenizer = self.tokenizer_reg
        self.visualize_training = False
        self.conv_temp_res = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.reg_loss_weight = loss_reg_weight
        
        
    def forward(self, res_only=False,generate_sent_only=False, **kwargs):
        torch.cuda.empty_cache()
        reg_outputs = self.model_reg.generate(
            kwargs['input_ids_reg'],
            images=kwargs['img_reg'],
            attention_mask= kwargs['attention_masks_reg'],
            labels=kwargs['labels_reg'],
            masks=kwargs['masks_list'], #prepared_reg_input['masks'], 
            use_cache=False
        )
        
        output_res = self.model_res(**kwargs) # output mask
        output_dict = {}
        # {"pred_masks": pred_masks, "gt_masks": masks_list, 'processed_masks':predicted_masks
        # add bbox loss
        if res_only:
            output_dict.update({'pred_masks': output_res['pred_masks'],'gt_masks': output_res['gt_masks']})
            return output_dict
        prepared_reg_input = self.prepare_reg_input_dict(output_res, **kwargs)
        
        reg_outputs = self.model_reg(
            kwargs['input_ids_reg'],
            images=kwargs['img_reg'],
            attention_mask= kwargs['attention_masks_reg'],
            labels=kwargs['labels_reg'],
            masks=kwargs['masks_list'], #prepared_reg_input['masks'], 
            use_cache=False
        )
        # import pdb;pdb.set_trace()
        # loss, logits
        if self.vis_tokens:
            return {'seg_token_feat':output_res['seg_token_feat'], 
                    'last_token_feat':reg_outputs['vis_token_per_layer']['last_token'], 
                    'last3_token_feat':reg_outputs['vis_token_per_layer']['last3_token'],
                    'mask_token_feat':reg_outputs['vis_token_per_layer']['mask_token'],
                    'pos_token_feat':reg_outputs['vis_token_per_layer']['pos_token'],
                    'vis_top20_feat':reg_outputs['vis_token_per_layer']['img_top20_token'],
                    }
        
        if generate_sent_only:
            output_dict.update({output_reg['output_sentence'], output_reg['output_sentence_w_gtbox']})
            return output_dict
        output_dict.update({"loss": output_res['loss_dict']['loss']+reg_outputs.loss* self.reg_loss_weight, 
                            'ce_loss_res': output_res['loss_dict']['ce_loss'], 
                            'mask_bce_loss': output_res['loss_dict']['mask_bce_loss'], 
                            'mask_dice_loss': output_res['loss_dict']['mask_dice_loss'], 
                            'mask_loss': output_res['loss_dict']['mask_loss'],
                            'ce_loss_reg': reg_outputs.loss* self.reg_loss_weight, 
                            'pred_masks': output_res['pred_masks'],
                            'gt_masks': output_res['gt_masks']})
        return output_dict
    

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps = self._get_batch_logps(all_logits, input_ids, attention_mask=att_masks, average_log_prob=False)
        chosen_logps = all_logps[: chosen_ids.shape[0]]
        rejected_logps = all_logps[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss
    
    
    