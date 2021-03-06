#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "codecov",
    "flake8",
    "black",
    "pytest",
    "pytest-cov",
    "pytest-raises",
]

setup_requirements = [
    "pytest-runner",
]

dev_requirements = [
    "bumpversion>=0.5.3",
    "coverage>=5.0a4",
    "flake8>=3.7.7",
    "ipython>=7.5.0",
    "m2r>=0.2.1",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "pytest-runner>=4.4",
    "Sphinx>=2.0.0b1",
    "sphinx_rtd_theme>=0.1.2",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

interactive_requirements = [
    "altair",
    "jupyterlab",
]

requirements = [
    "bokeh",
    "dask[bag]",
    "dask_jobqueue",
    "datastep>=0.1.5",
    "distributed>=2.11.0",
    "docutils<0.16",  # needed for botocore (quilt dependency)
    "fire",
    "matplotlib",
    "msgpack<1.0.0",
    "numpy",
    "pandas",
    "prefect",
    "python-dateutil<=2.8.0",  # need <=2.8.0 for quilt3 in step
    "seaborn",
]

extra_requirements = {
    "test": test_requirements,
    "setup": setup_requirements,
    "dev": dev_requirements,
    "interactive": interactive_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements,
    ],
}

setup(
    author="Jackson Maxfield Brown",
    author_email="jacksonb@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="An example step workflow that does a whole bunch of simple random matrix inversion.",
    entry_points={
        "console_scripts": ["example_step_workflow=example_step_workflow.bin.cli:cli"]
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="example_step_workflow",
    name="example_step_workflow",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="example_step_workflow/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/AllenCellModeling/example_step_workflow",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
