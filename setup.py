#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

version = {}
with open("spechomo/version.py") as version_file:
    exec(version_file.read(), version)

requirements = ['numpy', 'matplotlib', 'pandas', 'dill', 'nested_dict', 'tqdm', 'scipy', 'scikit-learn', 'geoarray',
                'seaborn', 'pyyaml', 'gms_preprocessing']

setup_requirements = []

test_requirements = ['coverage', 'nose', 'nose-htmloutput', 'rednose']

setup(
    author="Daniel Scheffler",
    author_email='daniel.scheffler@gfz-potsdam.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Spectral homogenization of multispectral satellite data.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='spechomo',
    name='spechomo',
    packages=find_packages(include=['spechomo']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/danschef/spechomo',
    version=version['__version__'],
    zip_safe=False,
)
