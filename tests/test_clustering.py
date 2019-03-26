#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_clustering
---------------

Tests for spechomo.clustering.KMeansRSImage
"""

import unittest
import os
import numpy as np
from sklearn.cluster import KMeans

from geoarray import GeoArray  # noqa E402 module level import not at top of file

from spechomo import __file__  # noqa E402 module level import not at top of file
from spechomo.clustering import KMeansRSImage  # noqa E402 module level import not at top of file

testdata = os.path.join(os.path.dirname(__file__), '../tests/data/Bavaria_farmland_LMU_Hyspex_subset.bsq')


class Test_KMeansRSImage(unittest.TestCase):
    """Tests class for spechomo.clustering.KMeansRSImage"""

    @classmethod
    def setUpClass(cls):
        cls.geoArr = GeoArray(testdata)
        cls.geoArr.to_mem()
        cls.kmeans = KMeansRSImage(cls.geoArr, n_clusters=10)

        os.environ['MPLBACKEND'] = 'Template'  # disables matplotlib figure popups # NOTE: import geoarray sets 'Agg'

    def test_compute_clusters(self):
        self.kmeans.compute_clusters()
        self.assertIsInstance(self.kmeans.clusters, KMeans)

    def test_apply_clusters(self):
        labels = self.kmeans.apply_clusters(self.geoArr)
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(labels.size == self.geoArr.rows * self.geoArr.cols)

    def test_get_random_spectra_from_each_cluster(self):
        random_samples = self.kmeans.get_random_spectra_from_each_cluster()
        self.assertIsInstance(random_samples, dict)
        for cluster_label in range(self.kmeans.n_clusters):
            self.assertIn(cluster_label, random_samples)

    def test_plot_cluster_centers(self):
        self.kmeans.plot_cluster_centers()

    def test_plot_cluster_histogram(self):
        self.kmeans.plot_cluster_histogram()

    def test_plot_clustered_image(self):
        self.kmeans.plot_clustered_image()
