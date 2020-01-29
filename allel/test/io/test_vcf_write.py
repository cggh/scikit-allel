# -*- coding: utf-8 -*-
import atexit
import os
import shutil
import tempfile
import warnings

import pytest
import zarr

from allel.io.vcf_read import vcf_to_zarr
from allel.io.vcf_write import write_vcf


# needed for PY2/PY3 consistent behaviour
warnings.resetwarnings()
warnings.simplefilter('always')


# setup temp dir for testing
tempdir = tempfile.mkdtemp()
atexit.register(shutil.rmtree, tempdir)


def fixture_path(fn):
    return os.path.join(os.path.dirname(__file__), os.pardir, 'data', fn)


@pytest.fixture(scope='module') # run once and used by all tests in this file
def zarr_callset():
    vcf_path = fixture_path('sample.vcf')
    zarr_path = os.path.join(tempdir, 'sample.zarr')
    vcf_to_zarr(vcf_path, zarr_path, fields='*')
    return zarr.open_group(zarr_path, mode='r')


def test_write_from_zarr_callset(zarr_callset):
    out_path = os.path.join(tempdir, 'out.vcf')
    write_vcf(out_path, zarr_callset)

    # TODO: Once the write function can write out full data,
    # modify the test so that it load back the written file and compare it with the original faithfully.
    random_line = '20	1110696	rs6040355	A	G,T'
    random_line = random_line.replace(r'\t', '\t')
    with open(out_path, 'r') as file:
        content = file.read()
        assert random_line in content
