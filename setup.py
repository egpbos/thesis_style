#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit thesis_style/__version__.py
version = {}
with open(os.path.join(here, 'thesis_style', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='thesis_style',
    version=version['__version__'],
    description="Matplotlib style for my PhD thesis",
    long_description=readme + '\n\n',
    author="E. G. Patrick Bos",
    author_email='egpbos@gmail.com',
    url='https://github.com/egpbos/thesis_style',
    packages=[
        'thesis_style',
    ],
    package_dir={'thesis_style':
                 'thesis_style'},
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='thesis_style',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn'
    ],
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
