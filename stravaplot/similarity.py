from itertools import tee
from typing import Iterable
from typing import Iterator
from typing import Tuple

import mmh3


def window(iterable: Iterable, size: int) -> Iterator[Tuple]:
    """ Create a moving window iterator over an iterable. """
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def common_prefix_len(a: str, b: str) -> int:
    """ Find the common prefix of two strings. """
    min_len = min(len(a), len(b))
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min_len


def b1_minhash_sig(shingles: Iterable, components: int = 256):
    """ Calculate a 1-bit minhash signature for a set of shingles. """
    sig = [
        min([mmh3.hash(shingle, seed, False) for shingle in shingles]) & 1
        for seed in range(components)
    ]
    return sig
