import pandas as pd
import pytest

import stravaplot.similarity as similarity


@pytest.mark.parametrize(
    'source,size,expected',
    [
        pytest.param([1, 2, 3, 4, 5], 3, [(1, 2, 3), (2, 3, 4), (3, 4, 5)], id='non-empty input'),
        pytest.param([], 2, [], id='empty input')
    ]
)
def test_window(source, size, expected):
    actual = similarity.window(source, size)
    assert list(actual) == expected


@pytest.mark.parametrize(
    'a,b,expected',
    [
        ('abdef', 'abc', 2),
        ('abc', 'abc', 3),
        ('abc', 'xyz', 0),
        ('abc', '', 0)
    ]
)
def test_common_prefix_len(a, b, expected):
    actual = similarity.common_prefix_len(a, b)
    assert actual == expected


@pytest.mark.parametrize(
    'raw,size,expected',
    [
        (pd.Series(['a', 'b', 'c', 'd']), 2, {'a b', 'b c', 'c d'}),
        (pd.Series(['a', 'b', 'b', 'a', 'b', 'c', 'd']), 3, {'a b a', 'b a b', 'a b c', 'b c d'})
    ]
)
def test_make_shingles(raw, size, expected):
    actual = similarity.make_shingles(raw, size)
    assert actual == expected
