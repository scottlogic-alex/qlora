# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence, TypedDict, List, Optional, Union, Literal, Tuple, OrderedDict
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch import LongTensor, FloatTensor
from torch.nn import Embedding, Linear
from contextlib import nullcontext
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BatchEncoding,
    set_seed,
    Seq2SeqTrainer,
    TrainerCallback,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from transformers.training_args import OptimizerNames
from datasets import load_dataset, Dataset, DatasetDict
# evaluate.py clashes with package 'evaluate'
sys.path.remove('')
import evaluate
sys.path.insert(0, '')

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from safetensors.torch import save_file
from src.gen_callback import GenerationCallback
from src.memory_usage_callback import MemoryUsageCallback
from src.terminate_callback import TerminateCallback
from src.collation import CollatedData, DataInstance
from src.truncation_side import truncation_side


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        logging.warning(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = "[BOS]"

process_supervision_tokens: Dict[str, str] = {
    'step_start':   '<|step_start|>',
    'step_end':     '<|step_end|>',
    'answer_start': '<|answer_start|>',
    'answer_end':   '<|answer_end|>',
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    lora_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Apply an existing LoRA as a starting point for finetuning"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf|prm800k-solutions]"}
    )
    register_process_supervision_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Register tokens for process supervision prompt template such as <|start_step|>, <|end_step|>"}
    )
    use_bos_token_in_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "If your model was pretrained to utilise BOS (e.g. LLaMA), then make use of it in prompt."}
    )
    register_bos_token: Optional[bool] = field(
        default=False,
        metadata={"help": "GPTNeoXTokenizer doesn't have a true BOS token registered. Register one."}
    )
    truncate_toward_center: Optional[bool] = field(
        default=False,
        metadata={"help": "Truncate prompt from left side, truncate continuation from right side."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
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
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Specify a checkpoint dir explicitly (this is a more precise way of specifying which checkpoint you want to resume from than output_dir, which would force you to pick the latest checkpoint from a given output dir)"}
    )
    optim: str = field(default=OptimizerNames.PAGED_ADAMW.value, metadata={"help": 'The optimizer to be used', 'choices': [e.value for e in OptimizerNames]})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": 'The eval batch size per GPU. Increase for better speed.'})
    evaluation_strategy: Literal['no', 'steps', 'epoch'] = field(default='no')
    eval_steps: Optional[int] = field(default=None)
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    save_safetensors: bool = field(default=False)
    torch_compile: bool = field(default=False)
    simulate_worst_case_seq_len: bool = field(default=False, metadata={"help": "pad prompts to maximum size, to help you measure the worst-case memory usage you'll experience in your dataset."})
    measure_memory: bool = field(default=False, metadata={"help": "Measures your VRAM usage at end of first step (i.e. after gradient accumulation)."})
    terminate_after_first_step: bool = field(default=False, metadata={"help": "shuts down Python without saving a checkpoint, after first step. This is to be used in concert with --measure_memory, so you can measure the step cost then kill the run."})
    metric_for_best_model: Optional[str] = field(default=None)
    torch_compile_mode: Optional[Literal['default', 'reduce-overhead', 'max-autotune']] = field(default=None)
    generate_steps: Optional[int] = field(default=None, metadata={"help": 'How frequently to test generation with a representative prompt (and report result)'})
    adapt_attn_only: bool = field(default=False, metadata={"help": 'Use original LoRA strategy of adapting only attn QKVO projections (i.e. not adapting MLPs). Not recommended, except for comparison purposes.'})
    quantize: bool = field(default=True, metadata={"help": 'Whether to use 4-bit/8-bit quantization. Disable this for comparison purposes only.'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        input_embedding: Embedding = kwargs["model"].get_input_embeddings()
        if any(map(lambda p: p.requires_grad, input_embedding.parameters())):
            state_dict: OrderedDict[str, FloatTensor] = input_embedding.state_dict()
            if args.save_safetensors:
                save_file(state_dict, os.path.join(checkpoint_folder, "embed_tokens.safetensors"))
            else:
                torch.save(state_dict, os.path.join(checkpoint_folder, "embed_tokens.pt"))
        
        lm_head: Optional[Linear] = kwargs["model"].get_output_embeddings()
        if lm_head is not None and any(map(lambda p: p.requires_grad, lm_head.parameters())):
            state_dict: OrderedDict[str, FloatTensor] = lm_head.state_dict()
            if args.save_safetensors:
                save_file(state_dict, os.path.join(checkpoint_folder, "lm_head.safetensors"))
            else:
                torch.save(state_dict, os.path.join(checkpoint_folder, "lm_head.pt"))

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir, lora_name_or_path: Optional[str] = None):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = {'': 0}

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK', '-1') != '-1':
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune or not args.quantize: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if args.quantize else None,
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast='pythia' in args.model_name_or_path or 'mpt' in args.model_name_or_path, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if args.use_bos_token_in_prompt:
        if tokenizer.bos_token_id == tokenizer.eos_token_id:
            assert args.register_bos_token, "Your BOS and EOS are the same. This is typical of models using GPTNeoXTokenizer, such as Pythia. you have expressed (via --use_bos_token_in_prompt) that you intend to use BOS in your prompt. This using BOS but having it mean EOS is probably not what you want. You should enable --register_bos_token, or disable --use_bos_token_in_prompt."
        if tokenizer._bos_token is None:
            assert args.register_bos_token, "You have required (via --use_bos_token_in_prompt) that BOS token be used in prompt, but this tokenizer doesn't have one. you should either register a BOS token via --register_bos_token, or (preferably) avoid using --use_bos_token_in_prompt, as the model was not pretrained to park attention on BOS when there's nothing to attend to. well, if you finetune it enough perhaps you'd get away with it."
    special_tokens: Dict[str, str] = {}
    if tokenizer._pad_token is None:
        special_tokens['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer._bos_token is None and args.register_bos_token:
        special_tokens['bos_token'] = DEFAULT_BOS_TOKEN
    if args.register_process_supervision_tokens:
        special_tokens['additional_special_tokens'] = list(process_supervision_tokens.values())
    needs_unfrozen_embed = special_tokens and special_tokens != { 'pad_token': DEFAULT_PAD_TOKEN }
    if special_tokens:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens,
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })
    
    if not args.full_finetune:
        if args.quantize:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
            # prepare_model_for_kbit_training() casts too many layers to float32, causing OOM.
            # it actually only intended to upcast norms and embeddings, but accidentally upcasts every Linear too.
            for name, param in model.named_parameters():
                # freeze base model's layers
                param.requires_grad = False
                if 'norm' in name:
                    # modeling_llama.LlamaRMSNorm
                    param.data = param.data.to(torch.float32)

            model.get_input_embeddings().weight.data = model.get_input_embeddings().weight.data.to(torch.float32)
            model.get_output_embeddings().weight.data = model.get_output_embeddings().weight.data.to(torch.float32)

            if args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                model.gradient_checkpointing_enable()

    if lora_name_or_path is not None:
        print(f"Loading base LoRA from checkpoint '{lora_name_or_path}'.")
        model = PeftModel.from_pretrained(
            model,
            lora_name_or_path,
            is_trainable=True,
        )

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print(f"Loading adapters from checkpoint '{checkpoint_dir}'.")
            model = PeftModel.from_pretrained(
                model,
                join(checkpoint_dir, 'adapter_model'),
                is_trainable=True,
            )
        else:
            print(f'adding LoRA modules...')
            modules: List[str] = [
                f'{p}_proj' for p in 'qkvo'
            ] if args.adapt_attn_only else find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
    
    if needs_unfrozen_embed:
        # PAD token is a special case where it's fine to add it yet not train it, because it'll be masked out anyway.
        # unfreezing embeddings can increase VRAM usage by several gigabytes for models such as Pythia, MPT or Falcon. it's less dramatic for LLaMA.
        print('Unfreezing embeddings, because you have registered additional non-PAD tokens. this will increase VRAM usage.')
        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool
    truncate_toward_center: bool
    use_bos_token_in_prompt: bool
    simulate_worst_case_seq_len: bool

    def __call__(self, instances: Sequence[DataInstance]) -> CollatedData:
        # Extract elements
        sources: List[str] = [f"{self.tokenizer.bos_token if self.use_bos_token_in_prompt else ''}{example['input']}" for example in instances]
        targets: List[str] = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        with truncation_side(self.tokenizer, 'left') if self.truncate_toward_center else nullcontext():
            tokenized_sources_with_prompt: BatchEncoding = self.tokenizer(
                sources,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False,
            )
        with truncation_side(self.tokenizer, 'right') if self.truncate_toward_center else nullcontext():
            tokenized_targets: BatchEncoding = self.tokenizer(
                targets,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )
        # Build the input and labels for causal LM
        input_ids: List[LongTensor] = []
        labels: List[LongTensor] = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                prompt_and_continuation: LongTensor = torch.tensor(tokenized_source + tokenized_target)
                # simulate worst-case sequence length
                if self.simulate_worst_case_seq_len:
                    prompt_and_continuation = pad(
                        prompt_and_continuation,
                        (0, (self.source_max_len + self.target_max_len) - (len(tokenized_source) + len(tokenized_target))),
                        mode='constant',
                        value=self.tokenizer.pad_token_id,
                    )
                input_ids.append(prompt_and_continuation)
                if not self.train_on_source:
                    source_ignored: LongTensor = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    # simulate worst-case sequence length
                    if self.simulate_worst_case_seq_len:
                        source_ignored = pad(
                            source_ignored,
                            (0, (self.source_max_len + self.target_max_len) - (len(tokenized_source) + len(tokenized_target))),
                            mode='constant',
                            value=self.tokenizer.pad_token_id,
                        )
                    labels.append(source_ignored)
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids: LongTensor = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels: Optional[LongTensor] = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict: CollatedData = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

class ExtractedPRM800KSolutionsSample(TypedDict):
    input: str
    output: str

class PRM800KSolutionsSample(TypedDict):
    instruction: str
    responses: List[str]
    next_response: str
    answer: Optional[str]

process_supervision_prompt = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

answer_response = '''{response}{answer}'''

def format_step(step: str) -> str:
    return ''.join((
        process_supervision_tokens['step_start'],
        step,
        process_supervision_tokens['step_end'],
    ))

def format_answer(answer: str) -> str:
    return ''.join((
        process_supervision_tokens['answer_start'],
        answer,
        process_supervision_tokens['answer_end'],
    ))

def format_answer_response(
    response: str,
    answer: str,
    answer_inband = True,
) -> str:
    """
    answer_inband=True:
    <|step_start|>Exactly.<|answer_start|>468<|answer_end|><|step_end|>

    answer_inband=False:
    <|step_start|>Exactly.<|step_end|><|answer_start|>468<|answer_end|>
    """
    formatted_answer: str = format_answer(answer)
    if answer_inband:
        return format_step(f'{response}{formatted_answer}')
    return f'{format_step(response)}{formatted_answer}'

def extract_prm800k_solutions_dataset(sample: PRM800KSolutionsSample) -> ExtractedPRM800KSolutionsSample:
    history: str = ''.join((format_step(response) for response in sample['responses']))
    latest_step: str = format_step(sample['next_response']) if sample['answer'] is None else format_answer_response(
        response = sample['next_response'],
        answer = sample['answer'],
    )
    output: str = f'{history}{latest_step}'
    mapped: ExtractedPRM800KSolutionsSample = {
        'input': process_supervision_prompt.format(instruction=sample['instruction']),
        'output': output,
    }
    return mapped

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args: argparse.Namespace) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples
        - prm800k-solutions, 12.4k examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name) -> Union[Dataset, DatasetDict]:
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'prm800k-solutions':
            return load_dataset("Birchlabs/openai-prm800k-solutions-only")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset: Union[Dataset, DatasetDict], dataset_format: Optional[str]) -> Union[Dataset, DatasetDict]:
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        elif dataset_format == 'prm800k-solutions':
            dataset = dataset.map(extract_prm800k_solutions_dataset, remove_columns=[
                'instruction',
                'responses',
                'next_response',
                'answer',
            ])
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if args.dataset_format == 'prm800k-solutions':
        assert args.truncate_toward_center is True, "prm800k-solutions dataset expects output to follow from input without a discontinuity, so enable --truncate_toward_center"

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
        truncate_toward_center=args.truncate_toward_center,
        use_bos_token_in_prompt=args.use_bos_token_in_prompt,
        simulate_worst_case_seq_len=args.simulate_worst_case_seq_len,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir) -> Tuple[Optional[str], bool]:
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments,
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    if training_args.checkpoint_dir is None:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
        if completed_training:
            print('Detected that training was already completed!')
    else:
        checkpoint_dir: str = training_args.checkpoint_dir
        print(f'checkpoint_dir {checkpoint_dir} was specified')
        completed_training = True

    model, tokenizer = get_accelerate_model(args, checkpoint_dir, model_args.lora_name_or_path)

    # clean up the buffers allocated by PeftModel.from_pretrained() in get_accelerate_model().
    # PEFT init allocates and deallocates a lot of memory.
    torch.cuda.empty_cache()

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    if training_args.report_to and 'wandb' in training_args.report_to:
        import wandb
        wandb.init(
            entity='sl-ml',
            project='llm-stepwise',
            name=f'qlora' if args.run_name is None else args.run_name,
            config={
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "bits": training_args.bits,
                "source_max_len": data_args.source_max_len,
                "source_max_len": data_args.source_max_len,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "optim": training_args.optim,
                "lora_r": training_args.lora_r,
                "lora_alpha": training_args.lora_alpha,
                "quant_type": training_args.quant_type,
                "double_quant": training_args.double_quant,
                "adam8bit": training_args.adam8bit,
                "full_finetune": training_args.full_finetune,
                "weight_decay": training_args.weight_decay,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "warmup_ratio": training_args.warmup_ratio,
                "torch_compile": training_args.torch_compile,
                "torch_compile_mode": training_args.torch_compile_mode,
            }
        )
    callbacks: List[TrainerCallback] = []
    if training_args.generate_steps is not None:
        assert args.dataset_format == 'prm800k-solutions', 'in-run continuation of representative prompts is only implemented for prm800k-solutions dataset_format'
        gen_callback = GenerationCallback(
            model=model,
            tokenizer=tokenizer,
            dataset=data_module['eval_dataset'],
            generation_config=training_args.generation_config,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            truncate_toward_center=args.truncate_toward_center,
            use_bos_token_in_prompt=args.use_bos_token_in_prompt,
            report_to_wandb=training_args.report_to and 'wandb' in training_args.report_to,
            generate_steps=training_args.generate_steps,
        )
        callbacks.append(gen_callback)
    if training_args.measure_memory:
        memory_usage_callback = MemoryUsageCallback()
        callbacks.append(memory_usage_callback)
    if training_args.terminate_after_first_step:
        terminate_callback = TerminateCallback()
        callbacks.append(terminate_callback)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
