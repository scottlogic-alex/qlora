from typing import TypedDict, Protocol, Sequence
from torch import LongTensor, BoolTensor
import sys
if sys.version_info < (3, 11):
	from typing_extensions import NotRequired
else:
	from typing import NotRequired

class DataInstance(TypedDict):
	input: str
	output: str

class CollatedData(TypedDict):
	input_ids: LongTensor
	attention_mask: BoolTensor
	labels: NotRequired[LongTensor]

class Collator(Protocol):
	def __call__(self, instances: Sequence[DataInstance]) -> CollatedData: ...