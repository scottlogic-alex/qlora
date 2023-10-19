from dataclasses import dataclass
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

@dataclass
class TerminateCallback(TrainerCallback):
    final_step: int
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.final_step:
            print(f'Reached step {self.final_step}; terminating due to TerminateCallback')
            # note: does not consider possibility of mixed-precision fp8. I can live with that.
            uses_half_precision = (args.fp16 or args.bf16) and args.half_precision_backend in ('auto', 'cuda')
            half_precision_dtype = 'float16' if args.fp16 else 'bfloat16'
            precision = f'mixed ({half_precision_dtype})' if uses_half_precision else 'uniform'
            print(f'''model:     {kwargs['model'].config._name_or_path}
precision: {precision}
grad acc:  {args.gradient_accumulation_steps}
grad ckpt: {args.gradient_checkpointing}
optim:     {args.optim.value}''')
            raise SystemExit(0)