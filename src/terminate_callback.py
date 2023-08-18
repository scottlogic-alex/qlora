from dataclasses import dataclass
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

@dataclass
class TerminateCallback(TrainerCallback):
    final_step: int
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.final_step:
            print(f'Reached step {self.final_step}; terminating due to TerminateCallback')
            raise SystemExit(0)