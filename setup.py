# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

DISTNAME = 'scikit-allel'
DESCRIPTION = 'A Python package for exploring and analysing genetic ' \
              'variation data'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Alistair Miles',
MAINTAINER_EMAIL = 'alimanfoo@googlemail.com',
URL = 'https://github.com/cggh/scikit-allel'
LICENSE = 'MIT'
DOWNLOAD_URL = URL
PACKAGE_NAME = 'allel'
EXTRA_INFO = dict(install_requires=['numpy'],
                  classifiers=['Development Status :: 1 - Planning',
                               'Intended Audience :: Developers',
                               'Intended Audience :: Science/Research',
                               'License :: OSI Approved :: MIT License',
                               'Topic :: Scientific/Engineering'])
