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
import os
from tempfile import TemporaryDirectory
from pandas import DataFrame

from spechomo.utils import (
    list_available_transformations,
    export_classifiers_as_JSON,
    download_pretrained_classifiers
)
from spechomo import __path__

classifier_rootdir = os.path.join(__path__[0], '..', 'tests', 'data', 'classifiers', 'SAMclassassignment')


class Test_Utils(unittest.TestCase):
    def test_list_available_transformations(self):
        trafoslist = list_available_transformations(classifier_rootDir=classifier_rootdir)

        self.assertIsInstance(trafoslist, DataFrame)
        self.assertTrue(len(trafoslist) != 0)

    def test_export_classifiers_as_JSON(self):
        with TemporaryDirectory() as td:
            export_classifiers_as_JSON(export_rootDir=td, method='LR', src_sat='Landsat-8', tgt_sat='Sentinel-2A',
                                       n_clusters=5, classifier_rootDir=classifier_rootdir)

            # negative n_clusters
            with self.assertWarns(RuntimeWarning):
                export_classifiers_as_JSON(export_rootDir=td, method='LR', src_sat='Landsat-8', tgt_sat='Sentinel-2A',
                                           n_clusters=-5, classifier_rootDir=classifier_rootdir)

            # QR is currently not supported
            with self.assertRaises(RuntimeError):
                export_classifiers_as_JSON(export_rootDir=td, method='QR', src_sat='Landsat-8', tgt_sat='Sentinel-2A',
                                           n_clusters=50, classifier_rootDir=classifier_rootdir)

    def test_download_pretrained_classifiers(self):
        with TemporaryDirectory() as td:
            self.assertIsNotNone(download_pretrained_classifiers('LR', os.path.join(td, 'not_existing_subdir')))
            self.assertIsNotNone(download_pretrained_classifiers('QR', os.path.join(td, 'not_existing_subdir')))
