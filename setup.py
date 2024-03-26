# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
import setuptools_scm


DISTNAME = "scikit-allel"

PACKAGE_NAME = "allel"

DESCRIPTION = "A Python package for exploring and analysing genetic variation data."

MAINTAINER = "Alistair Miles"

MAINTAINER_EMAIL = "alimanfoo@googlemail.com"

URL = "https://github.com/cggh/scikit-allel"

DOWNLOAD_URL = "http://pypi.python.org/pypi/scikit-allel"

LICENSE = "MIT"

INSTALL_REQUIRES = ["numpy", "dask[array]"]

# full installation with all optional dependencies
EXTRAS_REQUIRE = {
    "full": [
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "h5py",
        "numexpr",
        "zarr",
        "hmmlearn",
        "protopunica",
        "nose",
    ]
}

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]


# noinspection PyUnresolvedReferences
def setup_extensions(metadata):
    try:
        print("[scikit-allel] setup extensions with cython")
        from Cython.Build import cythonize
        import numpy

        ext_modules = cythonize(
            [
                Extension(
                    "allel.opt.model",
                    sources=["allel/opt/model.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    "allel.opt.stats",
                    sources=["allel/opt/stats.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    "allel.opt.io_vcf_read",
                    sources=["allel/opt/io_vcf_read.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
            ]
        )
        metadata["ext_modules"] = ext_modules
    except ImportError:
        print("[scikit-allel] cython not available, not including extensions")


def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        package_dir={"": "."},
        packages=find_packages(),
        package_data={"allel.test": ["data/*"]},
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        zip_safe=False,
    )
    setup_extensions(metadata)
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
