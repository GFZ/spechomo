#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_classifier
---------------

Tests for spechomo.classifier
"""

import os
from unittest import TestCase

from spechomo.classifier import Cluster_Learner
from spechomo import __path__

classifier_rootdir = os.path.join(__path__[0], 'resources', 'classifiers')


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
