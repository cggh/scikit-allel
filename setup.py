# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from ast import literal_eval
from setuptools import setup, Extension, find_packages
try:
    from Cython.setuptools import built_ext
except ImportError:
    from setuptools.command.build_ext import build_ext

DISTNAME = 'scikit-allel'

PACKAGE_NAME = 'allel'

DESCRIPTION = 'A Python package for exploring and analysing genetic ' \
              'variation data.'

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'Alistair Miles'

MAINTAINER_EMAIL = 'alimanfoo@googlemail.com'

URL = 'https://github.com/cggh/scikit-allel'

DOWNLOAD_URL = 'http://pypi.python.org/pypi/scikit-allel'

LICENSE = 'MIT'

INSTALL_REQUIRES = ['cython', 'numpy', 'dask[array]']

# full installation with all optional dependencies
EXTRAS_REQUIRE = {'full': ['scipy', 'matplotlib', 'seaborn', 'pandas', 'scikit-learn',
                           'h5py', 'numexpr', 'bcolz', 'zarr', 'hmmlearn',
                           'pomegranate', 'nose']}

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
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]


# noinspection PyUnresolvedReferences
def setup_extensions(metadata):
    # check for cython
    try:
        from Cython.Build import cythonize
        print('[scikit-allel] setup extensions with cython')
        ext_modules = cythonize([
            Extension('allel.opt.model',
                      sources=['allel/opt/model.pyx']
                      # define_macros=[('CYTHON_TRACE', 1)],
                      ),
            Extension('allel.opt.stats',
                      sources=['allel/opt/stats.pyx']
                      # define_macros=[('CYTHON_TRACE', 1)],
                      ),
            Extension('allel.opt.io_vcf_read',
                      sources=['allel/opt/io_vcf_read.pyx'],
                      # define_macros=[('CYTHON_TRACE', 1)],
                      ),
        ])
    except ImportError:
        print('[scikit-allel] setup extensions without cython')
        ext_modules = [
            Extension('allel.opt.model',
                      sources=['allel/opt/model.c']),
            Extension('allel.opt.stats',
                      sources=['allel/opt/stats.c']),
            Extension('allel.opt.io_vcf_read',
                      sources=['allel/opt/io_vcf_read.c'])
        ]
    metadata['ext_modules'] = ext_modules


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


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
        package_dir={'': '.'},
        packages=find_packages(),
        package_data={'allel.test': ['data/*']},
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        setup_requires=[
            'setuptools>18.0',
            'setuptools-scm>1.5.4'
        ],
        use_scm_version={
            'version_scheme': 'guess-next-dev',
            'local_scheme': 'dirty-tag',
            'write_to': 'allel/version.py'
        },
        zip_safe=False,
        cmdclass={'build_ext': CustomBuildExtCommand},
    )
    setup_extensions(metadata)
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
