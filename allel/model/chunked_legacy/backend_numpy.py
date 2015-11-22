# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from .backend_base import Backend


class NumpyBackend(Backend):
    """Reference implementation, will not be efficient."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create(self, data, expectedlen=None, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        return np.array(data, **kwargs)

    def append(self, arr, data):
        return np.append(arr, data, axis=0)

    def create_table(self, data, expectedlen=None, **kwargs):
        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)
        return np.rec.array(data, **kwargs)

    def append_table(self, arr, data):
        return np.append(arr, data)


# singleton instance
numpy_backend = NumpyBackend()


