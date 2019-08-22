#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019  Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz-potsdam.de)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

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
                'seaborn', 'pyyaml', 'tabulate', 'specclassify>=0.2.0', 'pyrsr']

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
