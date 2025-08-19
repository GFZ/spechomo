#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019-2021
# - Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz.de)
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
test_clustering
---------------

Tests for spechomo.clustering.KMeansRSImage
"""

import unittest
import os
import numpy as np

from geoarray import GeoArray

import spechomo
from spechomo.clustering import KMeansRSImage

testdata = os.path.join(os.path.dirname(spechomo.__file__), '../tests/data/AV_mastercal_testdata.bsq')


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
        from sklearn.cluster import KMeans  # avoids static TLS error here
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


if __name__ == '__main__':
    import pytest
    pytest.main()
