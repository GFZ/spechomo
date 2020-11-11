#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2020  Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz-potsdam.de)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. Please note the following exception: `spechomo` depends on tqdm,
# which is distributed under the Mozilla Public Licence (MPL) v2.0 except for the
# files "tqdm/_tqdm.py", "setup.py", "README.rst", "MANIFEST.in" and ".gitignore".
# Details can be found here: https://github.com/tqdm/tqdm/blob/master/LICENCE.
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

req = [
    'dill',
    'geoarray>=0.10.4',
    'matplotlib',
    'natsort',
    'nested_dict',
    'numpy',
    'pandas',
    'pyrsr',
    'pyyaml',
    'scikit-learn>=0.23.2',
    'scipy',
    'seaborn',
    'specclassify>=0.2.0',
    'tabulate',
    'tqdm',
]

req_setup = ['setuptools-git']  # needed for package_data version-controlled by GIT

req_test = ['coverage', 'nose', 'nose2', 'nose-htmloutput', 'rednose']

req_doc = ['sphinx-argparse', 'sphinx_rtd_theme', 'sphinx-autodoc-typehints']

req_lint = ['flake8', 'pycodestyle', 'pydocstyle']

req_dev = req_setup + req_test + req_doc + req_lint

setup(
    author="Daniel Scheffler",
    author_email='daniel.scheffler@gfz-potsdam.de',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Spectral homogenization of multispectral satellite data.",
    extras_require={
        "doc": req_doc,
        "test": req_test,
        "lint": req_lint,
        "dev": req_dev
    },
    install_requires=req,
    license="GPL-3.0-or-later",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    package_data={'spechomo': ['options_default.yaml',
                               # 'resources/**/**/*'
                               ]},
    keywords=['spechomo', 'spectral homogenization', 'sensor fusion', 'remote sensing'],
    name='spechomo',
    packages=find_packages(include=['spechomo'], exclude=['tests']),
    setup_requires=req_setup,
    test_suite='tests',
    tests_require=req_test,
    url='https://gitext.gfz-potsdam.de/geomultisens/spechomo',
    version=version['__version__'],
    zip_safe=False,
)
