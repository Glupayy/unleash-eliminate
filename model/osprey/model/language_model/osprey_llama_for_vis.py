#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

from ..osprey_arch import OspreyMetaModel, OspreyMetaForCausalLM

from ..layer import MaskExtractor
from typing import Any

class CustomCausalLMOutputWithPast(ModelOutput):
    
    # def __init__(self):
    #     super(CausalLMOutputWithPast, self).__init__()
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    vis_token_per_layer: Any = None

class OspreyConfig(LlamaConfig):
    model_type = "osprey"


class OspreyLlamaModel(OspreyMetaModel, LlamaModel):
    config_class = OspreyConfig

    def __init__(self, config: LlamaConfig):
        super(OspreyLlamaModel, self).__init__(config)


class OspreyLlamaForCausalLM(LlamaForCausalLM, OspreyMetaForCausalLM):
    config_class = OspreyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = OspreyLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mask_extractor = MaskExtractor()

        # Initialize weights and apply final processing
        self.post_init()
        self.vis_token = True

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_metas = None,
        masks = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CustomCausalLMOutputWithPast]:
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_token_len = input_ids.shape[1]
        image_start_token = []
        for one_sample in input_ids:
            image_flag = torch.nonzero(one_sample.squeeze() == -200).tolist()[0][0]
            image_start_token.append(image_flag)
            
        # import pdb;pdb.set_trace()
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, mask_t_mask, pos_t_mask = self.prepare_inputs_labels_for_multimodal(input_ids, masks, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.bfloat16()
  
        self.model = self.model.bfloat16()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict
        )

        # 看第一层之后的激活，不看embedding
        img_token_get20_idx = []
        for i,layer_hs in enumerate(outputs[1][1]):
            # import pdb;pdb.set_trace()
            img_hs = layer_hs[image_start_token[i]:image_start_token[i]+1024,:]
            hidden_dim = img_hs.shape[-1]
            # img_mean = img_hs.mean(dim=0, keepdim=True)
            # img_std = img_hs.std(dim=0, keepdim=True)
            # outlier_1v = torch.where((img_hs - (img_mean + 6 * img_std)) > 0.0, 1.0, 0.0) #
            # outlier_2v = torch.where((img_hs - (img_mean - 6 * img_std)) < 0.0, 1.0, 0.0) # 
            avg_token_feature_p = img_hs.mean(dim=0, keepdim=True)
            activation_p = (img_hs - avg_token_feature_p).norm(dim=-1)
            activation_mean = activation_p.mean(dim=0,keepdim=True)
            activation_std = activation_p.std(dim=0,keepdim=True)
            outlier_1a = torch.where((activation_p - (activation_mean + 6 * activation_std)) > 0.0, 1.0, 0.0)
            outlier_2a = torch.where((activation_p - (activation_mean - 6 * activation_std)) < 0.0, 1.0, 0.0)
            outliers_mask_p = outlier_1a + outlier_2a
            masked_mag_activation_p = activation_p.masked_fill(outliers_mask_p.bool(), float('-inf'))
            top_20_values_p, top_20_indices_p = torch.topk(masked_mag_activation_p.view(-1), 20)
            img_token_get20_idx.append(top_20_indices_p)
            # import pdb;pdb.set_trace()

#             avg_token_feature_p = image_features_projected.mean(dim=0, keepdim=True)
# acti        vation_p = (image_features_projected - avg_token_feature_p).norm(dim=-1)
# mag_        min_p, mag_max_p = activation_p.min(), activation_p.max()
# # print(mag_min,mag_max)
# mag_        activation_p = (activation_p - mag_min_p) / (mag_max_p - mag_min_p)
            # import pdb;pdb.set_trace()
        
        # first delete outliers of
        # self.get_image_token_mask()
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        hidden_states = outputs[0]
        self.lm_head = self.lm_head.to(hidden_states.dtype)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]

            return (loss,) + output if loss is not None else output

        if self.vis_token:
            mask_list_layers = []
            pos_list_layers = []
            last_token_layers = []
            last_3token_layres=[]
            image_top20_p_layer = []
            logits_all = []
            with torch.no_grad():
                # bbox_token_mask = self._create_bbox_token_mask(input_ids)
                # import pdb;pdb.set_trace()
                # no first embedding layer
                for idx_h in range(1, len(outputs[1])):
                    mask_i = outputs[1][idx_h][mask_t_mask].to(dtype=torch.float32).cpu().numpy()
                    mask_list_layers.append(mask_i)
                    pos_list_layers.append(outputs[1][idx_h][pos_t_mask].to(dtype=torch.float32).cpu().numpy())
                    last_token_layers.append(outputs[1][idx_h][:,-1].to(dtype=torch.float32).cpu().numpy())
                    last_3token_layres.append(outputs[1][idx_h][:,-3].to(dtype=torch.float32).cpu().numpy())
                    logits_all.append(self.lm_head(outputs[1][idx_h][:,-1]).to(dtype=torch.float32).cpu().numpy())
                    img_token_ = []
                    for batch_i in range(len(hidden_states)):
                        img_hs = outputs[1][idx_h][batch_i][image_start_token[batch_i]:image_start_token[batch_i]+1024,:]
                        # import pdb;pdb.set_trace()
                        img_token_.append(img_hs[img_token_get20_idx[batch_i]].unsqueeze(0))
                    image_top20_p_layer.append(torch.cat(img_token_,dim=0).to(dtype=torch.float32).cpu().numpy())
            # import pdb;pdb.set_trace()

        return CustomCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vis_token_per_layer={'last_token':last_token_layers, 'last3_token': last_3token_layres, 'mask_token':mask_list_layers,'pos_token':pos_list_layers,'img_top20_token':image_top20_p_layer,'logits_32':logits_all}
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("osprey", OspreyConfig)
AutoModelForCausalLM.register(OspreyConfig, OspreyLlamaForCausalLM)
