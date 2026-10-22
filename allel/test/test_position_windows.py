"""Regressions for inclusive physical-position window boundaries (#363)."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import allel


@pytest.mark.parametrize('size,step,expected', [
    (1, None, [[1, 1], [2, 2], [3, 3], [4, 4]]),
    (3, None, [[1, 3], [4, 4]]),
    (2, None, [[1, 2], [3, 4]]),
    (5, None, [[1, 4]]),
    (2, 1, [[1, 2], [2, 3], [3, 4]]),
    (2, 3, [[1, 2], [4, 4]]),
])
def test_inclusive_position_windows(size, step, expected):
    positions = [1, 2, 3, 4]
    counts, windows = allel.windowed_count(positions, size=size, step=step)
    assert_array_equal(windows, expected)
    assert_array_equal(counts, [right - left + 1 for left, right in expected])


@pytest.mark.parametrize('size', [1, 10])
def test_single_position_window(size):
    counts, windows = allel.windowed_count([17], size=size)
    assert_array_equal(windows, [[17, 17]])
    assert_array_equal(counts, [1])


@pytest.mark.parametrize('dtype', ['u1', 'i2', 'u4', 'i8', 'u8'])
@pytest.mark.parametrize('numpy_size', [False, True])
def test_position_windows_at_integer_limit(dtype, numpy_size):
    limit = np.iinfo(dtype).max
    positions = np.array([limit - 1, limit], dtype=dtype)
    size = positions.dtype.type(1) if numpy_size else 1
    counts, windows = allel.windowed_count(positions, size=size)
    expected = np.column_stack((positions, positions))
    assert_array_equal(windows, expected)
    assert_array_equal(counts, [1, 1])


def test_final_position_window_diversity():
    positions = [1, 2, 3, 4]
    counts = [[4, 0], [3, 1], [2, 2], [1, 3]]
    diversity, windows, bases, n_variants = allel.windowed_diversity(
        positions, counts, size=3)
    assert_array_equal(windows, [[1, 3], [4, 4]])
    # Four haplotypes give six unordered pairs: 0, 3, 4, 3 differ per site.
    np.testing.assert_allclose(diversity, [7 / 18, 1 / 2])
    assert_array_equal(bases, [3, 1])
    assert_array_equal(n_variants, [3, 1])


def test_inaccessible_final_position_window():
    diversity, windows, bases, n_variants = allel.windowed_diversity(
        [1, 2, 3, 4], np.array([[4, 0], [3, 1], [2, 2], [1, 3]]), size=3,
        start=1, stop=4,
        is_accessible=[True, True, True, False])
    assert_array_equal(windows, [[1, 3], [4, 4]])
    assert_array_equal(bases, [3, 0])
    assert_array_equal(n_variants, [3, 0])
    assert np.isnan(diversity[-1])
