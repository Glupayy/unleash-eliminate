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

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..osprey_arch import OspreyMetaModel, OspreyMetaForCausalLM

from ..layer import MaskExtractor

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
        self.tokenizer = None
        

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
        images: Optional[torch.FloatTensor] = None, # 1,3,512,512
        return_dict: Optional[bool] = None,
        which_layer: Optional[int] = None, #(0-31)
        xigao_generate_mode: Optional[bool] = False,
        # xigao_layers: Optional[List[int]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import pdb;pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_token_len = input_ids.shape[1]
        # import pdb;pdb.set_trace()
        
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, masks, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.half()
  
        # self.model = self.model.bfloat16()
        # if inputs_embeds is not None:
        #     inputs_embeds = inputs_embeds.float()
  
        # self.model = self.model.float()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
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

        output_dict = CausalLMOutputWithPast(
            loss=loss,
            logits=logits, # change to logit of all layers
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # import pdb;pdb.set_trace()
        return output_dict

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,**kwargs
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
