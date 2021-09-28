#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019-2021
# - Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz-potsdam.de)
# - Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam,
#   Germany (https://www.gfz-potsdam.de/)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
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
test_classifier
---------------

Tests for spechomo.classifier
"""

import os
import json
from unittest import TestCase
from tempfile import TemporaryDirectory

from spechomo.classifier import Cluster_Learner
from spechomo import __path__

# classifier_rootdir = os.path.join(__path__[0], 'resources', 'classifiers')
classifier_rootdir = os.path.join(__path__[0], '..', 'tests', 'data', 'classifiers', 'SAMclassassignment')


class Test_ClusterClassifier(TestCase):
    def setUp(self):
        self.clf = Cluster_Learner.from_disk(
            classifier_rootDir=classifier_rootdir,
            method='LR',
            n_clusters=3,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'])

    def test_plot_sample_spectra__single_cluster(self):
        self.clf.plot_sample_spectra(cluster_label=0)

    def test_plot_sample_spectra__selected_clusters(self):
        self.clf.plot_sample_spectra(cluster_label=[0, 2])

    def test_plot_sample_spectra__all_clusters(self):
        self.clf.plot_sample_spectra(cluster_label='all')

    def test_collect_stats(self):
        self.clf._collect_stats(cluster_label=1)

    def test_to_jsonable_dict(self):
        jsonable_dict = self.clf.to_jsonable_dict()
        outstr = json.dumps(jsonable_dict, sort_keys=True, indent=4)
        self.assertIsInstance(outstr, str)

    def test_save_to_json(self):
        with TemporaryDirectory() as tmpDir:
            self.clf.save_to_json(os.path.join(tmpDir, 'clf.json'))


# class Test_ClassifierCollection(TestCase):
#     def setUp(self) -> None:
#         self.CC = ClassifierCollection('/home/gfz-fe/scheffler/temp/SPECHOM_py/classifiers/'
#                                        '20k_nofilt_noaviris__SCADist100SAM4/LR_clust2__Sentinel-2A__MSI.dill')
#
#     def test_save_to_json(self):
#         with TemporaryDirectory() as tmpDir:
#             self.CC.save_to_json(os.path.join(tmpDir, 'CC.json'))
