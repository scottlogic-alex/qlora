from typing import Iterator, Iterable, TypeVar, Optional, Generator
from itertools import islice

T = TypeVar('T')

def nth(iterable: Iterable[T], n: int, default: Optional[T] = None) -> Optional[T]:
	"Returns the nth item or a default value"
	return next(islice(iterable, n, None), default)

def repeatedly(iterable: Iterable[T]) -> Generator[T, None, None]:
	while True:
		it: Iterator[T] = iter(iterable)
		yield from it