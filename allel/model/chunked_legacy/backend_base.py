# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from allel.model.ndarray import subset


class Backend(object):

    def subset(self, data, sel0, sel1, blen=None, **kwargs):

        # check inputs
        length = len(data)
        sel0 = np.asanyarray(sel0)
        sel1 = np.asanyarray(sel1)

        # ensure boolean array for dim 0
        if sel0.shape[0] < length:
            # assume indices, convert to boolean condition
            tmp = np.zeros(length, dtype=bool)
            tmp[sel0] = True
            sel0 = tmp

        # ensure indices for dim 1
        if sel1.shape[0] == data.shape[1]:
            # assume boolean condition, convert to indices
            sel1 = np.nonzero(sel1)[0]

        # block size for iteration
        if blen is None:
            blen = get_chunklen(data)

        # build output
        sel0_nnz = self.count_nonzero(sel0)
        out = None
        for i in range(0, length, blen):
            j = min(i+blen, length)
            bsel0 = sel0[i:j]
            # don't bother doing anything unless we have to
            n = np.count_nonzero(bsel0)
            if n:
                block = np.asanyarray(data[i:j])
                res = subset(block, bsel0, sel1)
                if out is None:
                    out = self.create(res, expectedlen=sel0_nnz, **kwargs)
                else:
                    out = self.append(out, res)

        return out

    def hstack(self, tup, blen=None, **kwargs):

        # check inputs
        if not isinstance(tup, (tuple, list)):
            raise ValueError('expected tuple or list, found %r' % tup)
        if len(tup) < 2:
            raise ValueError('expected two or more arrays to stack')

        def mapper(*blocks):
            return np.hstack(blocks)

        return self.map_blocks(tup, mapper, blen=blen, **kwargs)

    def vstack(self, tup, blen=None, **kwargs):

        # check inputs
        if not isinstance(tup, (tuple, list)):
            raise ValueError('expected tuple or list, found %r' % tup)
        if len(tup) < 2:
            raise ValueError('expected two or more arrays to stack')

        # set block size to use
        if blen is None:
            blen = min([get_chunklen(a) for a in tup])

        # build output
        expectedlen = sum(len(a) for a in tup)
        out = None
        for a in tup:
            for i in range(0, len(a), blen):
                j = min(i+blen, len(a))
                block = np.asanyarray(a[i:j])
                if out is None:
                    out = self.create(block, expectedlen=expectedlen, **kwargs)
                else:
                    out = self.append(out, block)
        return out

    def vstack_table(self, tup, blen=None, **kwargs):

        # check inputs
        if not isinstance(tup, (tuple, list)):
            raise ValueError('expected tuple or list, found %r' % tup)
        if len(tup) < 2:
            raise ValueError('expected two or more tables to stack')

        # set block size to use
        if blen is None:
            blen = min([get_chunklen_table(t) for t in tup])

        # build output
        expectedlen = sum(len(t) for t in tup)
        out = None
        for t in tup:
            for i in range(0, len(t), blen):
                j = min(i+blen, len(t))
                block = np.asanyarray(t[i:j])
                if out is None:
                    out = self.create_table(block, expectedlen=expectedlen,
                                            **kwargs)
                else:
                    out = self.append_table(out, block)
        return out

    def op_scalar(self, data, op, other, blen=None, **kwargs):

        # check inputs
        if not np.isscalar(other):
            raise ValueError('expected scalar')

        def mapper(block):
            return op(block, other)

        return self.map_blocks(data, mapper, blen=blen, **kwargs)

