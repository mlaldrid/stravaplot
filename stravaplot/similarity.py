from itertools import tee
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Set
from typing import Tuple

import mmh3
import pandas as pd
import pygeohash


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


# TODO try more shingling approaches
def make_shingles(raw: pd.Series, size: int) -> Set:
    """ Create shingles from the raw data. """
    no_repeats = raw[raw.shift() != raw]
    shingles = set([' '.join(group) for group in window(no_repeats, size)])
    return shingles


def b1_minhash_sig(shingles: Set, components: int = 256) -> List[int]:
    """ Calculate a 1-bit minhash signature for a set of shingles. """
    sig = [
        min([mmh3.hash(shingle, seed, False) for shingle in shingles]) & 1
        for seed in range(components)
    ]
    return sig


def geohash_track(track: pd.DataFrame, precision: int) -> pd.Series:
    """ Calculate geohash for each point in a track. """
    geohashes = track.apply(lambda pt: pygeohash.encode(pt['lat'], pt['lon'], precision=precision), axis=1)
    return geohashes
