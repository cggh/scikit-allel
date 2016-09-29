# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from ast import literal_eval
from setuptools import setup, Extension


DISTNAME = 'scikit-allel'

PACKAGE_NAME = 'allel'

DESCRIPTION = 'A Python package for exploring and analysing genetic ' \
              'variation data.'

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'Alistair Miles',

MAINTAINER_EMAIL = 'alimanfoo@googlemail.com',

URL = 'https://github.com/cggh/scikit-allel'

DOWNLOAD_URL = 'http://pypi.python.org/pypi/scikit-allel'

LICENSE = 'MIT'

# strictly speaking, allel requires numpy, scipy and numexpr, but numexpr
# won't install unless numpy is already installed, so leave this blank for now
# and require user to pre-install numpy, scipy and numexpr themselves
INSTALL_REQUIRES = []

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
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


# noinspection PyUnresolvedReferences
def setup_extensions(metadata):

    # TODO review this, numpy as optional dependency doesn't really work,
    # lots of stuff breaks if the C extensions don't get built

    try:
        # only build extensions if numpy is available
        import numpy as np
    except ImportError:
        # numpy not available
        pass
    else:

        # check for cython
        try:
            print('[scikit-allel] build with cython')
            from Cython.Build import cythonize
            ext_modules = cythonize([
                Extension('allel.opt.model',
                          sources=['allel/opt/model.pyx'],
                          include_dirs=[np.get_include()],
                          # define_macros=[('CYTHON_TRACE', 1)],
                          ),
                Extension('allel.opt.stats',
                          sources=['allel/opt/stats.pyx'],
                          include_dirs=[np.get_include()],
                          # define_macros=[('CYTHON_TRACE', 1)],
                          ),
            ])
        except ImportError:
            print('[scikit-allel] build from C')
            ext_modules = [
                Extension('allel.opt.model',
                          sources=['allel/opt/model.c'],
                          include_dirs=[np.get_include()]),
                Extension('allel.opt.stats',
                          sources=['allel/opt/stats.c'],
                          include_dirs=[np.get_include()]),
            ]
        metadata['ext_modules'] = ext_modules


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
        packages=['allel', 'allel.model', 'allel.chunked', 'allel.stats',
                  'allel.opt', 'allel.test'],
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
    setup_extensions(metadata)
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
