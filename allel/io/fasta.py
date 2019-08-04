# -*- coding: utf-8 -*-
import numpy as np


def write_fasta(path, sequences, names, mode='w', width=80):
    """Write nucleotide sequences stored as numpy arrays to a FASTA file.

    Parameters
    ----------

    path : string
        File path.
    sequences : sequence of arrays
        One or more ndarrays of dtype 'S1' containing the sequences.
    names : sequence of strings
        Names of the sequences.
    mode : string, optional
        Use 'a' to append to an existing file.
    width : int, optional
        Maximum line width.

    """

    # check inputs
    if isinstance(sequences, np.ndarray):
        # single sequence
        sequences = [sequences]
        names = [names]
    if len(sequences) != len(names):
        raise ValueError('must provide the same number of sequences and names')
    for sequence in sequences:
        if sequence.dtype != np.dtype('S1'):
            raise ValueError('expected S1 dtype, found %r' % sequence.dtype)

    # force binary mode
    mode = 'ab' if 'a' in mode else 'wb'

    # write to file
    with open(path, mode=mode) as fasta:
        for name, sequence in zip(names, sequences):
            # force bytes
            if isinstance(name, str):
                name = name.encode('ascii')
            header = b'>' + name + b'\n'
            fasta.write(header)
            for i in range(0, sequence.size, width):
                line = sequence[i:i+width].tostring() + b'\n'
                fasta.write(line)
