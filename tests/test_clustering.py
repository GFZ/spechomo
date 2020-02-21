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
        cls.geoArr = GeoArray(testdata, nodata=-9999)
        cls.geoArr.to_mem()
        cls.geoArr[:10, :10, :10] = -9999  # simulate some pixels that have nodata in some bands (unusable for KMeans)

        cls.kmeans = KMeansRSImage(cls.geoArr, n_clusters=5,
                                   sam_classassignment=False)

        os.environ['MPLBACKEND'] = 'Template'  # disables matplotlib figure popups # NOTE: import geoarray sets 'Agg'

    def test_compute_clusters(self):
        self.kmeans.compute_clusters(nmax_spectra=1e5)
        self.assertIsInstance(self.kmeans.clusters, KMeans)

    def test_apply_clusters(self):
        labels = self.kmeans.apply_clusters(self.geoArr)
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(labels.size == self.geoArr.rows * self.geoArr.cols)

    def test_spectral_angles(self):
        angles = self.kmeans.spectral_angles
        self.assertIsInstance(angles, np.ndarray)

    def test_spectral_distances(self):
        distances = self.kmeans.spectral_distances
        self.assertIsInstance(distances, np.ndarray)

    def test_get_random_spectra_from_each_cluster(self):
        random_samples = self.kmeans.get_random_spectra_from_each_cluster()
        self.assertIsInstance(random_samples, dict)
        for cluster_label in range(self.kmeans.n_clusters):
            self.assertIn(cluster_label, random_samples)

        random_samples = self.kmeans.get_random_spectra_from_each_cluster(max_distance='50%', max_angle=4)
        self.assertIsInstance(random_samples, dict)
        for cluster_label in range(self.kmeans.n_clusters):
            self.assertIn(cluster_label, random_samples)

    def test_get_purest_spectra_from_each_cluster(self):
        random_samples = self.kmeans.get_purest_spectra_from_each_cluster()
        self.assertIsInstance(random_samples, dict)
        for cluster_label in range(self.kmeans.n_clusters):
            self.assertIn(cluster_label, random_samples)

    def test_plot_cluster_centers(self):
        self.kmeans.plot_cluster_centers()

    def test_plot_cluster_histogram(self):
        self.kmeans.plot_cluster_histogram()

    def test_plot_clustered_image(self):
        self.kmeans.plot_clustermap()
