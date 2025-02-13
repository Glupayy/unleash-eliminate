"""
train.py - GLaMM Training on Single Dataset Type

Trains the GLaMM model on one dataset type (Caption, Region, or Segmentation) at a time, iterating thoroughly through
the chosen dataset. This targeted approach is optimal for specialized training on specific downstream task.
"""
import os, re
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from typing import Dict, Optional, Sequence
from dataset.dataset_org import custom_collate_fn_ospreyeval_cyc
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU)

# from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
# from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
# from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
# from dataset.region_datasets.RefCOCO_VG_Region_ds import (RefCocoRegDataset, RefCocoGRegDataset, RefCocoPRegDataset,
#                                                        VisualGenomeRegDataset)
from dataset.segm_datasets.RefCOCO_Segm_ds import RESDataset_Osprey_RES_REG
from model.GLaMM_cyc_Osprey import GLaMMForCausalLMCycle_RES_REG
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from model.llava.mm_utils import tokenizer_image_token
from PIL import Image

def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Training")

    # Need to set the model and data paths
   #-----------------
    parser.add_argument("--version_res", default="GLAMM-RefSeg")
    parser.add_argument("--version_reg", default="Osprey-7b")
    parser.add_argument("--vision_pretrained", default="sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision_tower", default="clip-vit-large-patch14-336", type=str)
    
    parser.add_argument("--refcoco_image", default="coco_2014/train2014", type=str)
    parser.add_argument("--anno_path", default="finetune_refcocog_val_with_mask.json", type=str)
   #-----------------
    
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)


    parser.add_argument("--layer_s", default=1, type=int)
    parser.add_argument("--layer_e", default=11, type=int)
    parser.add_argument("--sample_n_layers", default=4, type=int)
    
    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--dataset_dir", default="GLAMM_data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str, help="Choose from: CocoCap, LLaVaInstruct")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=1, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=3e-6, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--ce_loss_weight_reg", default=1.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="RefCOCOgRegVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--val_dataset_reg", default=None, type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)
    parser.add_argument("--idx_num", default=0, type=int)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version_res, model_max_length=args.model_max_length, padding_side="right", use_fast=False,add_prefix_space=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version_res) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    # import pdb;pdb.set_trace()

    tokenizer_reg = transformers.AutoTokenizer.from_pretrained(
            args.version_reg,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
    tokenizer_reg.pad_token = tokenizer_reg.unk_token
    return tokenizer, tokenizer_reg


def initialize_model(args, tokenizer, tokenizer_reg):
    """ Initialize the GLaMM model. """
    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default="Osprey-7b")
        version: Optional[str] = field(default="v1")
        freeze_backbone: bool = field(default=False)
        tune_mm_mlp_adapter: bool = field(default=False)
        vision_tower: Optional[str] = field(default=None)
        mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
        pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
        mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
        mm_use_im_start_end: bool = field(default=False)
        mm_use_im_patch_token: bool = field(default=False)
        mm_vision_select_feature: Optional[str] = field(default="patch")


    @dataclass
    class TrainingArguments(transformers.TrainingArguments):
        cache_dir: Optional[str] = field(default=None)
        optim: str = field(default="adamw_torch")
        remove_unused_columns: bool = field(default=False)
        freeze_mm_mlp_adapter: bool = field(default=False)
        model_max_length: int = field(
            default=2048,
            metadata={
                "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            },
        )
        double_quant: bool = field(
            default=True,
            metadata={"help": "Compress the quantization statistics through double quantization."}
        )
        quant_type: str = field(
            default="nf4",
            metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
        )
        bits: int = field(
            default=16,
            metadata={"help": "How many bits to use."}
        )
        lora_enable: bool = True
        lora_r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        lora_weight_path: str = ""
        lora_bias: str = "none"
        group_by_modality_length: bool = field(default=False)
        output_dir: str = ''
    
    model_args = ModelArguments()
    training_args = TrainingArguments()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}

    model_args_res = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args_res["num_level_reg_features"] = 4  
    model = GLaMMForCausalLMCycle_RES_REG(args.version_res, args.version_reg, model_args_res,tokenizer,tokenizer_reg,args.ce_loss_weight_reg)
    
    print('\033[92m' + "---- Initialized RES model from: {} ----".format(args.version_res) + '\033[0m')
    print('\033[92m' + "---- Initialized REG model from: {} ----".format(args.version_reg) + '\033[0m')


    # Configure model tokens
    model.model_res.config.eos_token_id = tokenizer.eos_token_id
    model.model_res.config.bos_token_id = tokenizer.bos_token_id
    model.model_res.config.pad_token_id = tokenizer.pad_token_id

    return model, model_args, training_args

def prepare_model_for_training_reg(model, tokenizer, training_args, model_args):
    # Enable input gradients
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=args.local_rank)

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    def find_all_linear_names_2(model):
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, linear_cls) and all(
                x not in name for x in ['vision_tower','mm_projector','mask_extractor']
            ) and any(x in name for x in ['q_proj','v_proj']):
                lora_module_names.add(name)
                # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(list(lora_module_names))
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names_2(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    set_trainable_modules_reg(model)


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.float16, device=args.local_rank)

    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)

def set_trainable_modules_reg(model):
    trainable_modules = ["lm_head", "embed_tokens", 'mask_extractor', 'mm_projector','model.norm']
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_names = len([name for name, param in model.named_parameters() if param.requires_grad])
        print('\033[92m' + "---- Total parameters Osprey: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters Osprey: ----{}".format(trainable_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters name number Osprey: ----{}".format(total_names) + '\033[0m')
        
    count_parameters(model)

def initialize_datasets_and_loaders(args, tokenizer,tokenizer_reg):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}


    world_size = torch.cuda.device_count()
    val_datasets_reg=[]
    # Summing lengths of all datasets
    if args.val_dataset_reg:
            val_datasets_reg.append(RESDataset_Osprey_RES_REG(
                annotation_file=args.anno_path,
                root_path = args.refcoco_image,
                tokenizer = tokenizer_reg,
                image_size= args.image_size,
                image_encoder_path=args.vision_tower
            ))
    else:
        val_datasets_reg = None
    return val_datasets_reg


def setup_data_loaders(args, val_datasets_reg,tokenizer,tokenizer_reg):
    spi_tokens = ['<mask>', '<pos>']
    tokenizer_reg.add_tokens(spi_tokens, special_tokens=True)
   
    inference_mode = True
    
    collate_fn_args_val_reg = custom_collate_fn_ospreyeval_cyc

    if val_datasets_reg:
        # import pdb;pdb.set_trace()
        combined_val_datasets_reg = ConcatDataset(val_datasets_reg)
        val_loader_reg = torch.utils.data.DataLoader(
            combined_val_datasets_reg, batch_size=1, num_workers=0, collate_fn=collate_fn_args_val_reg,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets_reg, rank=args.local_rank, shuffle=False), )
    return val_loader_reg


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8, "offload_optimizer":{"device": "cpu"}}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )

    return model_engine, optimizer, scheduler


def main(args):
    tokenizer, tokenizer_reg = setup_tokenizer_and_special_tokens(args)
    # import pdb;pdb.set_trace()
    model, model_args, training_args = initialize_model(args, tokenizer,tokenizer_reg)
    prepare_model_for_training(model.model_res, tokenizer, args)
    prepare_model_for_training_reg(model.model_reg, tokenizer_reg, training_args, model_args)
    
    val_datasets_reg = initialize_datasets_and_loaders(args, tokenizer,tokenizer_reg)
    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    
    # val_loader: a list of val dataloader 
    val_loader_reg = setup_data_loaders(args, val_datasets_reg, tokenizer, tokenizer_reg)
    # dataset_iter = iter(train_loader)

    writer = initialize_environment(args)
    
    # eval_only:
    validate_model_performance_reg(val_loader_reg, model_engine, writer, args)
    exit()

def validate_model_performance_reg(val_loader_reg, model_engine, writer, args):
    captions_all = []
    captions_persam = []
    all_temp_caption = []

    dict_eval = {}
    model_engine.eval()
    model_reg = model_engine.model_reg
    model_res = model_engine.model_res
    
    end_idx = None
    if args.idx_num!=0 :
        end_idx = args.idx_num
    for idx, (input_ids, image, masks, stop_str,img_id,global_enc_image,grounding_enc_image,resize_list, original_size_list) in enumerate(tqdm.tqdm(val_loader_reg)):
        
        if idx>=10:
            break
        layer_candidate = range(args.layer_s,args.layer_e) 
        select_layers = random.sample(layer_candidate, args.sample_n_layers) 
        print('-----Candidate Layer {}-----'.format(layer_candidate))
        print('-----Selected Layer {}-----'.format(select_layers))
        temp_caption = []
        for layer_ in select_layers:
            caption = {}
            caption['assert'] = None
            with torch.no_grad():
                model_reg.orig_forward = model_reg.forward
                model_reg.forward = partial(model_reg.orig_forward,
                                                img_metas=[None],
                                                masks=[masks[0].half()])

                output_ids = model_reg.generate(
                        input_ids[0],
                        images=image[0].unsqueeze(0).cuda(),
                        max_new_tokens=50,
                        use_cache=False,
                        num_beams=1,
                        cycd_layers=[layer_],
                        layers_prob=None
                )

                model_reg.forward = model_reg.orig_forward

            input_token_len = input_ids[0].shape[1]
            n_diff_input_output = (
                input_ids[0] != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = model_reg.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                    skip_special_tokens=True)[0]

            outputs = outputs.strip()
            if outputs.endswith(stop_str[0]):
                outputs = outputs[:-len(stop_str[0])]
            outputs = outputs.strip().lower()
            if ':' in outputs:
                outputs = outputs.split(':')[1]
            
            prompt_res_org = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: The <im_start><image><im_end> provides an overview of the picture.\n"
            
            prompt_res_org += "Could you provide a segmentation mask for the {} in this image? ASSISTANT:".format(outputs)
            
            global_enc_image = global_enc_image[0].unsqueeze(0).cuda().half()
            # post_h, post_w = global_enc_image.shape[1:3]
            bboxes = None
            input_ids_res = tokenizer_image_token(prompt_res_org, model_engine.tokenizer_res, return_tensors="pt")
            input_ids_res = input_ids_res.unsqueeze(0).cuda() 
            grounding_enc_image = grounding_enc_image[0].unsqueeze(0).cuda()
            grounding_enc_image = grounding_enc_image.half()
            
            try:
                output_ids_res, pred_masks, mask_score = model_res.evaluate(
                    global_enc_image, grounding_enc_image, input_ids_res, resize_list, original_size_list, max_tokens_new=512,
                bboxes=bboxes)
                mask_score=mask_score.to(torch.float32).cpu().item()
            
            # print(mask_score)
                assert len(pred_masks)!= 0 and pred_masks != None , "no pred_masks!!"
            except AssertionError as e:
                caption['assert'] = f'{idx} sample in {layer_} does not have the pred_masks'
                captions_all.append(caption)
                temp_caption.append(caption)
                continue

            predicted_masks = (pred_masks[0] > 0).int()

            # for idx,mask_i in enumerate(predicted_masks):
            #     mask_i = mask_i.cpu().numpy()*255
            #     mask_i = mask_i.astype('uint8')
            #     mask_i = Image.fromarray(mask_i)
            #     # import pdb;pdb.set_trace()
            #     mask_i.save('./mask_'+str(idx)+'_'+str(mask_score[idx].float().cpu().numpy())+'.jpg')
            
            
            mean_iou = calculate_iou(predicted_masks[0], masks[0]).to(torch.float32).cpu().numpy().tolist()
            # model_reg
            # print(outputs)
            outputs = outputs.replace('.', '.\n')
            caption['image_id'] = img_id[0]
            caption['caption'] = outputs
            caption['layer'] = layer_

            caption['sam_score'] = mask_score
            # caption['dice_loss'] = dice_loss
            caption['iou'] = mean_iou

            
            print(caption)
            captions_all.append(caption)
            temp_caption.append(caption)
        # claculating the regular_score and save in all_temp_caption
        sam_total_iou = sum(item['iou'] for item in temp_caption)
        for item in temp_caption:
            item['regular_iou'] = item['iou']/sam_total_iou
        for item in temp_caption:
            item['regular_score'] = item['regular_iou'] * item['sam_score']
        all_temp_caption.append(temp_caption)

        # chose the bigest regular_score from the temp_caption
        max_dict = max(temp_caption, key=lambda x: x['regular_score'])
        captions_persam.append(max_dict)

        if end_idx != None and idx >= args.idx_num:
            break
    caption_file_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(caption_file_dir, exist_ok=True)

    results_path = f"{caption_file_dir}/caption_all_perlayer_{args.local_rank}.json"
    with open(results_path, 'w') as json_file:
        json.dump(captions_all, json_file, indent=2)
    # eval on rank 0

    results_path = f"{caption_file_dir}/caption_persam_{args.local_rank}.json"
    with open(results_path, 'w') as json_file:
        json.dump(captions_persam, json_file, indent=2)

    results_path = f"{caption_file_dir}/all_temp_caption_{args.local_rank}.json"
    with open(results_path, 'w') as json_file:
        json.dump(all_temp_caption, json_file, indent=2)

    torch.distributed.barrier()
    return 0

def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    import pdb;pdb.set_trace()

    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss

def calculate_iou(prediction, target):
    
    intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone().unsqueeze(0), target.contiguous(), 2, ignore_index=255)
  
    accuracy_iou = intersect / (union_ + 1e-5)
    # handles no-object targets
    accuracy_iou[union_ == 0] += 1.0
    accuracy_iou.cpu().numpy()
    return torch.mean(accuracy_iou)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
