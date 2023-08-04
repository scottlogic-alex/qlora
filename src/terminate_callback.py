from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class TerminateCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('Terminating on step end, due to TerminateCallback')
        raise SystemExit(0)