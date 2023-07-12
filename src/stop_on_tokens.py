from transformers import StoppingCriteria
from typing import List
from dataclasses import dataclass
from torch import LongTensor, FloatTensor

@dataclass
class StopOnTokens(StoppingCriteria):
	stop_token_ids: List[int]
	def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
		for stop_id in self.stop_token_ids:
			if input_ids[0][-1] == stop_id:
				return True
		return False