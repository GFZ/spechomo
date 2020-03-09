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

"""
test_utils
----------

Tests for spechomo.utils
"""

import unittest
from tempfile import TemporaryDirectory
from pandas import DataFrame

from spechomo.utils import list_available_transformations, export_classifiers_as_JSON


class Test_Utils(unittest.TestCase):
    def test_list_available_transformations(self):
        trafoslist = list_available_transformations()

        self.assertIsInstance(trafoslist, DataFrame)
        self.assertTrue(len(trafoslist) != 0)

    def test_export_classifiers_as_JSON(self):
        with TemporaryDirectory() as td:
            export_classifiers_as_JSON(export_rootDir=td, method='LR', src_sat='Landsat-8', tgt_sat='Sentinel-2A',
                                       n_clusters=5)

        with self.assertWarns(RuntimeWarning):
            export_classifiers_as_JSON(export_rootDir=td, method='LR', src_sat='Landsat-8', tgt_sat='Sentinel-2A',
                                       n_clusters=-5)

        # QR is currently not supported
        with self.assertRaises(RuntimeError):
            export_classifiers_as_JSON(export_rootDir=td, method='QR', src_sat='Landsat-8', tgt_sat='Sentinel-2A',
                                       n_clusters=5)
