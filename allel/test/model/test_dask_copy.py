"""Keep the typed lazy array API and independent annotations when copying."""
import dask.callbacks
import numpy as np
import pytest

import allel


@pytest.mark.parametrize('name,shape', [
    ('GenotypeDaskArray', (4, 3, 2)),
    ('GenotypeDaskVector', (4, 2)),
    ('HaplotypeDaskArray', (4, 6)),
    ('AlleleCountsDaskArray', (4, 3)),
    ('GenotypeAlleleCountsDaskArray', (4, 3, 3)),
    ('GenotypeAlleleCountsDaskVector', (4, 3)),
])
def test_copy_retains_dask_array_type(name, shape):
    cls = getattr(allel, name)
    values = np.arange(np.prod(shape), dtype='i2').reshape(shape) % 3
    original = cls(values, chunks=(2,) + shape[1:])
    executed = []
    with dask.callbacks.Callback(pretask=lambda *args: executed.append(args)):
        copied = original.copy()
    assert not executed
    assert type(copied) is cls
    assert copied.values is not original.values
    assert copied.chunks == original.chunks
    original.values[0] = 1
    np.testing.assert_array_equal(copied.values.compute(), values)


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('phased', [False, True])
def test_copy_retains_genotype_annotations(vector, masked, phased):
    cls = allel.GenotypeDaskVector if vector else allel.GenotypeDaskArray
    shape = (4, 2) if vector else (4, 3, 2)
    values = np.arange(np.prod(shape), dtype='i2').reshape(shape) % 3
    original = cls(values, chunks=(2,) + shape[1:])
    annotation = np.arange(np.prod(shape[:-1])).reshape(shape[:-1]) % 2 == 0
    if masked:
        original.mask = annotation
    if phased:
        original.is_phased = annotation

    copied = original.copy()

    assert type(copied) is cls
    if masked:
        assert copied.mask is not original.mask
        original.mask[0] = ~annotation[0]
        np.testing.assert_array_equal(copied.mask.compute(), annotation)
    else:
        assert copied.mask is None
    if phased:
        assert copied.is_phased is not original.is_phased
        original.is_phased[0] = ~annotation[0]
        np.testing.assert_array_equal(copied.is_phased.compute(), annotation)
    else:
        assert copied.is_phased is None
