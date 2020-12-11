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
test_classifier
---------------

Tests for spechomo.classifier
"""

import os
import json
from unittest import TestCase
# from tempfile import TemporaryDirectory

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

    # def test_save_to_json(self):
    #     with TemporaryDirectory() as tmpDir:
    #         self.clf.save_to_json(os.path.join(tmpDir, 'clf.json'))


# class Test_ClassifierCollection(TestCase):
#     def setUp(self) -> None:
#         self.CC = ClassifierCollection('/home/gfz-fe/scheffler/temp/SPECHOM_py/classifiers/'
#                                        '20k_nofilt_noaviris__SCADist100SAM4/LR_clust2__Sentinel-2A__MSI.dill')
#
#     def test_save_to_json(self):
#         with TemporaryDirectory() as tmpDir:
#             self.CC.save_to_json(os.path.join(tmpDir, 'CC.json'))
