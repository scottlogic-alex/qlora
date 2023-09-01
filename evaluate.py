from dataclasses import dataclass, field
from typing import Optional, TypedDict, NamedTuple, List, Dict, TypeAlias
import torch
from torch import LongTensor, FloatTensor
from torch.nn import Embedding, Linear
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  HfArgumentParser,
  set_seed,
  StoppingCriteriaList,
  LlamaForCausalLM,
  LlamaTokenizer,
  LlamaTokenizerFast
)
from peft import PeftModel, PeftModelForCausalLM
import logging
from enum import Enum
import sys
from os.path import splitext
from time import perf_counter

from transformers import AutoTokenizer
from safetensors.torch import load_file
from typing import Optional, OrderedDict, Literal

from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer
from src.stop_on_tokens import StopOnTokens

logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = "[BOS]"

process_supervision_tokens: Dict[str, str] = {
  'step_start':   '<|step_start|>',
  'step_end':     '<|step_end|>',
  'answer_start': '<|answer_start|>',
  'answer_end':   '<|answer_end|>',
}

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

class Participant(Enum):
  User = 'user'
  Assistant = 'assistant'
  System = 'system'

class Message(NamedTuple):
  participant: Participant
  message: str

class PromptStyle(Enum):
  Bare = 'bare'
  Chat = 'chat'
# I am not proud of this, but when I attempted to specify Enum fields on the arg dataclasses:
# hfparser.parse_args_into_dataclasses() turned the enum instances into string values.
# so we make some types to capture what we're actually going to receive.
PromptStyleLiteral: TypeAlias = Literal['bare', 'chat']

class Dtype(Enum):
  Bf16 = 'bf16'
  Fp16 = 'fp16'
  Fp32 = 'fp32'
DtypeLiteral: TypeAlias = Literal['bf16', 'fp16', 'fp32']

class SufficientResponse(BaseException): ...

def reify_dtype(dtype: DtypeLiteral) -> torch.dtype:
  match(dtype):
    case 'bf16':
      return torch.bfloat16
    case 'fp16':
      return torch.float16
    case 'fp32':
      return torch.float32

@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
    default="huggyllama/llama-7b"
  )
  use_flash_llama: Optional[bool] = field(
    default=False,
    metadata={"help": "Loads LLaMA models via flash attn 2."}
  )
  tokenizer_model_name_or_path: Optional[str] = field(
    default="huggyllama/llama-7b"
  )
  base_lora_model_name_or_path: Optional[str] = field(
    default=None,
    metadata={"help": "If you are a evaluating a LoRA of a LoRA (for example you have finetuned Alpaca): you can specify 'tloen/alpaca-lora-7b' here, then specify your downstream LoRA at via --lora_model_name_or_path."}
  )
  lora_model_name_or_path: Optional[str] = field(
    default=None,
    metadata={"help": "Example: tloen/alpaca-lora-7b. Apply this over the model (after applying base_lora_model_name_or_path, if specified)"}
  )
  input_embedding_path: Optional[str] = field(
    default=None,
    metadata={"help": "Pickle file containing the model's input embedding layer. i.e. the embed_tokens.pt which qlora.py will save out, if you retrained the embedding (the embedding layer becomes unfrozen if you expand its vocabulary)."}
  )
  output_embedding_path: Optional[str] = field(
    default=None,
    metadata={"help": "Pickle file containing the model's output embedding layer. i.e. the lm_head.pt which qlora.py will save out, if you retrained the lm_head (the lm_head becomes unfrozen if you expand its vocabulary)."}
  )
  trust_remote_code: Optional[bool] = field(
    default=False,
    metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
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
  model_dtype: DtypeLiteral = field(
    default=Dtype.Fp16.value,
    metadata={"help": "Compute type of the model. Used for non-quantized computations.", "choices": [p.value for p in Dtype]}
  )
  bnb_compute_dtype: DtypeLiteral = field(
    default=Dtype.Fp16.value,
    metadata={"help": "Compute type used for quantized computations. Prefer to turn this on if you are quantizing and your GPU supports it. You probably also want it even if you're not quantizing. Float16 should be better than bfloat16. Float32 can be slightly better than float16.", "choices": [p.value for p in Dtype]}
  )
  use_bos_token_in_prompt: Optional[bool] = field(
    default=False,
    metadata={"help": "If your model was pretrained to utilise BOS (e.g. LLaMA), then make use of it in prompt."}
  )

@dataclass
class MiscArguments:
  seed: Optional[int] = field(
    default=64,
    metadata={"help": "Random seed, for deterministic generation."}
  )
  compile: bool = field(
    default=False,
    metadata={"help": "Invoke torch.compile() on the model, with mode='max-autotune'. Requires PyTorch 2, CUDA, and either Python 3.10 or Python 3.11 with a recent torch nightly. Will make the first inference from the model take a bit longer, but subsequent inferences will be faster."}
  )
  system_prompt: str = field(
    default="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    metadata={"help": "The context which precedes the chat history. Can be used to influence the chatbot's responses."}
  )
  overrun_countermeasures: bool = field(
    default=True,
    metadata={"help": "Detect when bot is about to start talking to itself; end the generation before that happens. The bot is *supposed* to emit an end-of-sentence token to indicate that it's finished its reply, but neglects to do so in longer conversations, continuing to sequence-complete both sides of the conversation. Hence this countermeasure tries to detect and prevent that."}
  )
  tokenizer_cache_dir: Optional[str] = field(
    default=None
  )
  initial_input: Optional[str] = field(
    default=None,
    metadata={"help": "Initial message sent to the model. For example: What is $\sqrt{53}$ in simplest radical form?"}
  )
  # if you actually set the type hint to PromptStyle: you will find that HF/argparse assign a string anyway
  prompt_style: PromptStyleLiteral = field(
    default=PromptStyle.Chat.value,
    metadata={"choices": [p.value for p in PromptStyle]}
  )
  chat_memory: bool = field(
    default=False,
    metadata={"help": "Whether chat sequence should accumulate a conversation context, or reset each time"}
  )
  reseed_each_prompt: bool = field(
    default=True,
    metadata={"help": "Reset seed before each user input"}
  )
  show_seed: bool = field(
    default=True,
    metadata={"help": "Show seed in prompt"}
  )
  measure_perf: bool = field(
    default=True,
    metadata={"help": "Print inference speed"}
  )

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

def get_model(args: ModelArguments) -> LlamaForCausalLM:
  config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=args.trust_remote_code,
  )
  if args.use_flash_llama and config.model_type == 'llama':
    updates: Dict[str, Union[str, int, float, bool, None]] = {}
    flash_model_name = 'sl-alex/flash_llama--modeling_flash_llama.LlamaForCausalLM'
    if 'num_key_value_heads' not in config.__dict__:
      updates['num_key_value_heads'] = config.num_attention_heads
    if 'auto_map' in config.__dict__:
      if not ('AutoModelForCausalLM' in config.auto_map and 'flash' in config.auto_map['AutoModelForCausalLM']):
        updates['auto_map']['AutoModelForCausalLM'] = flash_model_name
    else:
      updates['auto_map'] = { 'AutoModelForCausalLM': flash_model_name }
    if 'rope_scaling' not in config.__dict__:
      updates['rope_scaling'] = { 'factor': (args.source_max_len + args.target_max_len)/config.max_position_embeddings, 'type': 'linear' }
    if 'pretraining_tp' not in config.__dict__:
      updates['pretraining_tp'] = 1
    if updates:
      config.update(updates)
  cuda_avail = torch.cuda.is_available()
  load_in_4bit = args.bits == 4 and cuda_avail
  load_in_8bit = args.bits == 8 and cuda_avail

  bnb_compute_dtype: torch.dtype = reify_dtype(args.bnb_compute_dtype)

  quantization_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=bnb_compute_dtype,
    bnb_4bit_use_double_quant=args.double_quant,
    bnb_4bit_quant_type=args.quant_type,
  ) if cuda_avail else None

  if not cuda_avail:
    logger.warning("You don't have CUDA, so we have turned off quantization. If you happen to be on a Mac: you probably have enough unified memory to run in fp16 anyway…")

  # Actually float16 supposedly has lower error than bfloat16 for *inference*.
  # if compute_dtype == torch.float16 and cuda_avail and torch.cuda.is_bf16_supported():
  #   print("Your GPU supports bfloat16; you may want to try it with --bf16 (note: I'm not sure how important this is for inference, but it's certainly preferred when training with 4-bit quantization.)")

  model_dtype: torch.dtype = reify_dtype(args.model_dtype)
  
  model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map=0,
    quantization_config=quantization_config,
    torch_dtype=model_dtype,
    trust_remote_code=args.trust_remote_code,
  ).eval()
  model.config.torch_dtype=model_dtype

  return model

def main():
  hfparser = HfArgumentParser((ModelArguments, GenerationArguments, MiscArguments))
  model_args, generation_args, misc_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
  if extra_args:
    raise ValueError(f"Received unsupported command-line args: {extra_args}")
  generation_config = GenerationConfig(**vars(generation_args))

  model: LlamaForCausalLM = get_model(model_args)

  tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(
    model_args.tokenizer_model_name_or_path or model_args.model_name_or_path,
    cache_dir=misc_args.tokenizer_cache_dir,
    use_fast = False,
    tokenizer_type='llama' if 'llama' in (model_args.tokenizer_model_name_or_path or model_args.model_name_or_path) else None, # Needed for HF name change
  )
  if 'falcon' in model_args.tokenizer_model_name_or_path:
    generation_config.eos_token_id = generation_config.pad_token_id = tokenizer.eos_token_id

  if 'llama' in model_args.tokenizer_model_name_or_path or isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
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

  if model_args.base_lora_model_name_or_path is not None:
    print(f'Applying base LoRA {model_args.base_lora_model_name_or_path}.')
    model: PeftModelForCausalLM = PeftModel.from_pretrained(
      model,
      model_args.base_lora_model_name_or_path,
    ).eval()

  if model_args.lora_model_name_or_path is not None:
    print(f'Applying LoRA {model_args.lora_model_name_or_path}.')
    model: PeftModelForCausalLM = PeftModel.from_pretrained(
      model,
      model_args.lora_model_name_or_path,
    ).eval()

  if model_args.input_embedding_path is None:
    learned_embed_count: int = model.get_input_embeddings().weight.shape[0]
    recognised_vocab_count: int = len(tokenizer)
    vocab_difference: int = recognised_vocab_count - learned_embed_count
    if vocab_difference == 1 and next(iter(tokenizer.get_added_vocab().values())) == tokenizer.pad_token_id:
      logger.info("Tokenizer knows more tokens than the model has learned embeddings for, but the difference is just an unlearned PAD token. So long as we stick to batches-of-1 or mask out the padding: we should be fine.")
    elif vocab_difference != 0:
      raise ValueError(f"must have an embedding per token in the tokenizer. tokenizer had {recognised_vocab_count} tokens, embedding had {learned_embed_count} embeddings.")
  else:
    print(f'Applying finetuned input embedding weights from {model_args.input_embedding_path}.')
    _, extension = splitext(model_args.input_embedding_path)
    if extension == '.pt':
      input_embed_state_dict: OrderedDict[str, FloatTensor] = torch.load(model_args.input_embedding_path, map_location=model.device, weights_only=True)
    else:
      assert extension == '.safetensors', "only .pt and .safetensors embeddings state dict files are supported"
      input_embed_state_dict: OrderedDict[str, FloatTensor] = load_file(model_args.input_embedding_path, device=model.device)
    embed_tokens: Embedding = model.get_input_embeddings()
    orig_device, orig_dtype = embed_tokens.weight.device, embed_tokens.weight.dtype

    assert input_embed_state_dict['weight'].shape[0] == len(tokenizer), f"embeddings state dict must have an embedding per token in the tokenizer. tokenizer had {len(tokenizer)} tokens, embedding had {input_embed_state_dict['weight'].shape[0]} embeddings."
    embed_tokens: Embedding = model.resize_token_embeddings(len(tokenizer))

    embed_tokens.load_state_dict(input_embed_state_dict)
    embed_tokens.weight.to(device=orig_device, dtype=orig_dtype)

  if model_args.output_embedding_path is not None:
    print(f'Applying finetuned output embedding weights from {model_args.output_embedding_path}.')
    _, extension = splitext(model_args.output_embedding_path)
    if extension == '.pt':
      lm_head_state_dict: OrderedDict[str, FloatTensor] = torch.load(model_args.output_embedding_path, map_location=model.device, weights_only=True)
    else:
      assert extension == '.safetensors', "only .pt and .safetensors embeddings state dict files are supported"
      lm_head_state_dict: OrderedDict[str, FloatTensor] = load_file(model_args.output_embedding_path, device=model.device)
    lm_head: Optional[Linear] = model.get_output_embeddings()
    assert lm_head is not None, "if you are finetuning the lm_head, it'd be weird to find that the base model didn't have one at all"
    orig_device, orig_dtype = lm_head.weight.device, lm_head.weight.dtype

    lm_head.load_state_dict(lm_head_state_dict)
    lm_head.weight.to(device=orig_device, dtype=orig_dtype)

  if misc_args.compile:
    torch.compile(model, mode='max-autotune')

  # stop_token_ids: List[int] = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
  stop_token_ids: List[int] = [tokenizer.eos_token_id, *tokenizer.convert_tokens_to_ids([process_supervision_tokens['answer_end']])]
  stop = StopOnTokens(stop_token_ids)
  stopping_criteria=StoppingCriteriaList([stop])

  history: List[Message] = [Message(Participant.System, misc_args.system_prompt)] if misc_args.system_prompt else []

  reset_ansi='\x1b[0m'
  cyan_ansi='\x1b[31;36m'
  blue_ansi='\x1b[31;34m'
  green_ansi='\x1b[31;32m'
  purple_ansi='\x1b[31;35m'

  participant_names: Dict[Participant, str] = {
    Participant.User: 'Instruction',
    Participant.Assistant: 'Response',
  }

  def format_message(envelope: Message) -> str:
    participant, message = envelope
    if participant is Participant.System:
      return message
    return f'### {participant_names[participant]}:\n{message}'
  
  next_seed: Optional[int] = None

  first = True
  while True:
    seed: int = misc_args.seed if next_seed is None else next_seed
    if misc_args.reseed_each_prompt or first or next_seed is not None:
      set_seed(seed)

    try:
      prompt_ctx: str = f'[seed={seed}]' if misc_args.show_seed else ''
      if first and misc_args.initial_input is not None:
        user_input = misc_args.initial_input
        quote: str = f'{purple_ansi}{prompt_ctx}> '
        print(f'{quote}{user_input}')
      else:
        prompt: str = f'{purple_ansi}{prompt_ctx}$ '
        user_input = input(f'{blue_ansi}Type a message to begin the conversation…{reset_ansi}\n{prompt}' if first else prompt)
    except KeyboardInterrupt:
      sys.exit(0)
    print(reset_ansi, end='')

    # you can 
    if user_input.startswith('!'):
      command, *rest = user_input[1:].split(' ')
      match command:
        case 'seed':
          assert len(rest) == 1, '"seed" command only takes one operand'
          operand, *_ = rest
          next_seed = int(operand)
          continue
        case _:
          raise ValueError(f'Command "{command}" not recognised. Recognised commands: seed')

    first = False
  
    match misc_args.prompt_style:
      case PromptStyle.Chat.value:
        chat_to_complete: str = '\n\n'.join([
          format_message(message) for message in [
            *history,
            Message(Participant.User, user_input),
            Message(Participant.Assistant, process_supervision_tokens['step_start']),
          ]
        ])
        if model_args.use_bos_token_in_prompt:
          chat_to_complete: str = f'{tokenizer.bos_token}{chat_to_complete}'
      case PromptStyle.Bare.value:
        chat_to_complete: str = user_input
      case _:
        raise ValueError(f'Never heard of a {misc_args.prompt_style} PromptStyle.')

    tokenized_prompts: TokenizerOutput = tokenizer([chat_to_complete], return_tensors='pt', truncation=True)
    
    print(green_ansi, end='', flush=True)

    response = ''
    if misc_args.overrun_countermeasures:
      # the model may continue adding to the conversation (replying to itself) instead of emitting an EOS token.
      # we try to intercept this. If it looks like it's starting a new message in the voice of either of the chat participants: don't print that, and stop generation.
      acc_overrun = ''

      def on_text(message: str, stream_end = False):
        nonlocal response, acc_overrun

        overrun_and_message = f'{acc_overrun}{message}'

        newline_ix = overrun_and_message.find('\n')
        if newline_ix > -1:
          pre_newline = overrun_and_message[:newline_ix]
          newline_onward = overrun_and_message[newline_ix:]

          if newline_onward.startswith('\n\n###'):
            raise SufficientResponse()
          if newline_onward.rstrip('\n\n###') == '':
            # could potentially grow into a \n\n###. Don't print it to the console just yet. we need to accumulate to see whether the bot's about to talk to itself.
            acc_overrun = newline_onward
            response += pre_newline
            print(pre_newline, end='', flush=True)
            return
          # newline_onward cannot grow into an Instruction/Response header, so this must be something else. flush everything we accumulated.

        response += overrun_and_message
        print(overrun_and_message, end='', flush=True)
        acc_overrun = ''
    else:
      def on_text(message: str, stream_end = False):
        nonlocal response
        response += message
        print(message, end='', flush=True)

    streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=False)

    try:
      inference_start: float = perf_counter()
      prediction: LongTensor = model.generate(
        input_ids=tokenized_prompts.input_ids.to(model.device),
        attention_mask=tokenized_prompts.attention_mask.to(model.device),
        generation_config=generation_config,
        do_sample=generation_config.temperature > 0.,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
      )
      # reset ANSI control sequence (plus line break)
      print(reset_ansi)
      # if you wanted to see the result, you can do so like this:
      # decode: List[str] = tokenizer.decode(prediction[0,tokenized_prompts.input_ids.size(-1):], skip_special_tokens=False, clean_up_tokenization_spaces=True)
      # print(decode)
      # pass
      # but we're already streaming it to the console via our callback
      inference_duration: float = perf_counter()-inference_start
      token_in_count: int = tokenized_prompts.input_ids.size(-1)
      token_out_count: int = prediction.size(-1) - token_in_count
      tokens_out_per_sec: float = token_out_count/inference_duration
      if misc_args.measure_perf:
        print(f'{cyan_ansi}ctx length: {token_in_count}\ntokens out: {token_out_count}\nduration: {inference_duration:.2f} secs\nspeed: {tokens_out_per_sec:.2f} tokens/sec{reset_ansi}')
    except (KeyboardInterrupt, SufficientResponse):
      # reset ANSI control sequence (plus line break)
      print(reset_ansi)

    if misc_args.chat_memory:
      history += [Message(Participant.Assistant, response)]

if __name__ == "__main__":
  main()