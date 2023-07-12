from typing import TypedDict
from torch import LongTensor, BoolTensor
import sys
if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
else:
    from typing import NotRequired

class CollatedData(TypedDict):
    input_ids: LongTensor
    attention_mask: BoolTensor
    labels: NotRequired[LongTensor]