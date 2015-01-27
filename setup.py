# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from ast import literal_eval


DISTNAME = 'scikit-allel'

PACKAGE_NAME = 'allel'

DESCRIPTION = 'A Python package for exploring and analysing genetic ' \
              'variation data.'

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'Alistair Miles',

MAINTAINER_EMAIL = 'alimanfoo@googlemail.com',

URL = 'https://github.com/cggh/scikit-allel'

DOWNLOAD_URL = 'http://pypi.python.org/pypi/scikit-allele'

LICENSE = 'MIT'

INSTALL_REQUIRES = ['numpy']

CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'License :: OSI Approved :: MIT License',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
]


def get_version(source='allel/__init__.py'):
    with open(source) as sf:
        for line in sf:
            if line.startswith('__version__'):
                return literal_eval(line.split('=')[-1].lstrip())
    raise ValueError("__version__ not found")

VERSION = get_version()


def setup_package():

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        version=VERSION,
        package_dir={'': '.'},
        packages=['allele', 'allele.test'],
        classifiers=CLASSIFIERS,
    )

    try:
        from setuptools import setup
        metadata['install_requires'] = INSTALL_REQUIRES

    except ImportError:
        from distutils.core import setup

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
