from transformers import (
  AutoConfig,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  AutoModelForCausalLM,
  LlamaTokenizerFast,
  PreTrainedModel,
  TextIteratorStreamer,
  StoppingCriteria,
  StoppingCriteriaList,
)
from typing import Dict, Union, TypedDict, Optional
from torch import LongTensor, FloatTensor
import torch
from time import perf_counter

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

reset_ansi='\x1b[0m'
green_ansi='\x1b[31;32m'
purple_ansi='\x1b[31;35m'
cyan_ansi='\x1b[31;36m'

# llama 1 models such as huggyllama/llama-7b work too
# model_name = 'huggyllama/llama-7b'
model_name = 'meta-llama/Llama-2-7b-chat-hf'
config = AutoConfig.from_pretrained(model_name)

use_flash_llama = True
if use_flash_llama and config.model_type == 'llama':
  updates: Dict[str, Union[str, int, float, bool, None]] = {}
  # this is a fork of togethercomputer/LLaMA-2-7B-32K's modeling_flash_llama.py, with a padding fix
  # https://huggingface.co/sl-alex/flash_llama/blob/main/modeling_flash_llama.py
  flash_model_name = 'sl-alex/flash_llama--modeling_flash_llama.LlamaForCausalLM'
  if 'auto_map' in config.__dict__:
    if not ('AutoModelForCausalLM' in config.auto_map and 'flash' in config.auto_map['AutoModelForCausalLM']):
      updates['auto_map']['AutoModelForCausalLM'] = flash_model_name
  else:
    updates['auto_map'] = { 'AutoModelForCausalLM': flash_model_name }
  # modeling_flash_llama.py expects some llama 2 config to be present. if this is a llama 1 model: we add the missing config
  if 'num_key_value_heads' not in config.__dict__:
    updates['num_key_value_heads'] = config.num_attention_heads
  if 'rope_scaling' not in config.__dict__:
    # if you want to finetune to a non-native context length, here's where you'd override it
    # updates['rope_scaling'] = { 'factor': context_length/config.max_position_embeddings, 'type': 'linear' }
    updates['rope_scaling'] = None
  if 'pretraining_tp' not in config.__dict__:
    updates['pretraining_tp'] = 1
  if updates:
    config.update(updates)

load_in_4bit=True
load_in_8bit=False

quantization_config: Optional[BitsAndBytesConfig] = BitsAndBytesConfig(
  load_in_4bit=load_in_4bit,
  load_in_8bit=load_in_8bit,
  llm_int8_threshold=6.0,
  llm_int8_has_fp16_weight=False,
  bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type='nf4',
) if load_in_4bit or load_in_8bit else None

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  load_in_4bit=load_in_4bit,
  load_in_8bit=load_in_8bit,
  device_map=0,
  quantization_config=quantization_config,
  torch_dtype=torch.float16,
  # "trust remote code" required because that's how we load modeling_flash_llama.py
  trust_remote_code=True,
  # Llama 2 requires accepting terms & conditions
  use_auth_token=True,
).eval()

tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
  model_name,
  use_fast=True,
  tokenizer_type='llama',
)

prompt = 'What the world needs now is'
tokenized: TokenizerOutput = tokenizer([prompt], return_tensors='pt', truncation=True)

print(f'{purple_ansi}> {prompt}{reset_ansi}')
colour_changed = False

class Streamer(TextIteratorStreamer):
  def on_finalized_text(self, text: str, stream_end: bool = False):
    # messy, but if we were to change console colour too early: warnings would get coloured the same way as model output
    global colour_changed
    if not colour_changed:
      print(green_ansi, end='', flush=True)
      colour_changed = True
    print(text, end='', flush=True)

class StopOnEOS(StoppingCriteria):
  def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
    return input_ids[0][-1] == tokenizer.eos_token_id
stopping_criteria = StoppingCriteriaList([StopOnEOS()])

try:
  inference_start: float = perf_counter()
  prediction: LongTensor = model.generate(
    input_ids=tokenized.input_ids.to(model.device),
    attention_mask=tokenized.attention_mask.to(model.device),
    generation_config=GenerationConfig(
      max_new_tokens=200,
    ),
    do_sample=True,
    stopping_criteria=stopping_criteria,
    streamer=Streamer(tokenizer, skip_prompt=True),
  )
  # reset ANSI control sequence (plus line break)
  print(reset_ansi)
  # if you wanted to see the result, you can do so like this:
  # decode: List[str] = tokenizer.decode(prediction[0,tokenized.input_ids.size(-1):], skip_special_tokens=False, clean_up_tokenization_spaces=True)
  # but we're already streaming it to the console via our callback
  inference_duration: float = perf_counter()-inference_start
  token_in_count: int = tokenized.input_ids.size(-1)
  token_out_count: int = prediction.size(-1) - token_in_count
  tokens_out_per_sec: float = token_out_count/inference_duration
  print(f'{cyan_ansi}ctx length: {token_in_count}\ntokens out: {token_out_count}\nduration: {inference_duration:.2f} secs\nspeed: {tokens_out_per_sec:.2f} tokens/sec{reset_ansi}')
except (KeyboardInterrupt):
  print(reset_ansi)