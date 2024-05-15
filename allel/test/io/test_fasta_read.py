import os

from allel.io.fasta import read_fasta


def fixture_path(fn):
    return os.path.join(os.path.dirname(__file__), os.pardir, 'data', fn)


def test_fasta_read():
    path = fixture_path('test.fa')
    data = read_fasta(path)
    assert len(data.keys()) == 9


def test_fasta_read_comments():
    path = fixture_path('test_comments.fa')
    data = read_fasta(path)
    assert len(data.keys()) == 2
    assert 'sequence A' in data.keys()
    assert 'sequence B' in data.keys()
