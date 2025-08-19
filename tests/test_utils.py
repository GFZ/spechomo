#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019-2021
# - Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz.de)
# - GFZ Helmholtz Centre for Geosciences, Potsdam, Germany (https://www.gfz.de)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Please note the following exception: `spechomo` depends on tqdm, which is
# distributed under the Mozilla Public Licence (MPL) v2.0 except for the files
# "tqdm/_tqdm.py", "setup.py", "README.rst", "MANIFEST.in" and ".gitignore".
# Details can be found here: https://github.com/tqdm/tqdm/blob/master/LICENCE.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


if __name__ == '__main__':
    import pytest
    pytest.main()
