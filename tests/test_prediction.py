#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019  Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz-potsdam.de)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
test_prediction
---------------

Tests for spechomo.prediction
"""


import unittest
import os
import numpy as np
from geoarray import GeoArray

from spechomo.prediction import SpectralHomogenizer, RSImage_ClusterPredictor
from spechomo.utils import spectra2im
from spechomo import __path__

classifier_rootdir = os.path.join(__path__[0], 'resources', 'classifiers')
testdata_rootdir = os.path.join(__path__[0], '..', 'tests', 'data')


class Test_SpectralHomogenizer(unittest.TestCase):
    """Tests class for spechomo.SpectralHomogenizer"""

    @classmethod
    def setUpClass(cls):
        cls.SpH = SpectralHomogenizer(classifier_rootDir=classifier_rootdir)
        cls.testArr_L8 = GeoArray(np.random.randint(1, 10000, (50, 50, 7), dtype=np.int16))  # no band 9, no pan
        # cls.testArr_L8 = GeoArray('/home/gfz-fe/scheffler/temp/SPECHOM_py/images_train/conserv/'
        #                           # 'Landsat-8_OLI_TIRS__f141006t01p00r16_refl_preprocessed.bsq')
        #                           'Landsat-8_OLI_TIRS__f130502t01p00r09_refl_master_calibration_Aviris_'
        #                           'preprocessed.bsq')
        # cls.testArr_L8 = GeoArray('/home/gfz-fe/scheffler/temp/'
        #                           'Landsat-8__OLI_TIRS__LC81940242014072LGN00_L2B__250x250.bsq')  # no pan
        # cls.testArr_L8 = GeoArray('/home/gfz-fe/scheffler/temp/'
        #                           'Landsat-8__OLI_TIRS__LC81940242014072LGN00_L2B.bsq')  # no pan
        # cls.testArr_L8 = GeoArray('/home/gfz-fe/scheffler/temp/'
        #                           'clusterhomo_sourceL8_full_withoutB9.bsq')  # no pan, no cirrus

        cls.cwl_L8 = [442.98, 482.59, 561.33, 654.61, 864.57, 1609.09, 2201.25]
        # cls.cwl_L8 = [442.98, 482.59, 561.33, 654.61, 864.57, 1373.48, 1609.09, 2201.25]

    def test_interpolate_cube_linear(self):
        outarr = self.SpH.interpolate_cube(self.testArr_L8, self.cwl_L8, [500., 700., 1300.], kind='linear')
        self.assertIsInstance(outarr, np.ndarray)
        self.assertEqual(outarr.shape, (50, 50, 3))
        self.assertEqual(outarr.dtype, np.int16)

    def test_interpolate_cube_quadratic(self):
        outarr = self.SpH.interpolate_cube(self.testArr_L8, self.cwl_L8, [500., 700., 1300.], kind='quadratic')
        self.assertIsInstance(outarr, np.ndarray)
        self.assertEqual(outarr.shape, (50, 50, 3))
        self.assertEqual(outarr.dtype, np.int16)

    def test_predict_by_machine_learner__LR_L8_S2(self):
        """Test linear regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='LR', n_clusters=50,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
            compute_errors=True,
            classif_alg='SAM',
            src_nodataVal=-9999
        )

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 11))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 11))
        self.assertEqual(errors.dtype, np.int16)

    def test_predict_by_machine_learner__LR_RF_L8_S2(self):
        """Test linear regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='LR', n_clusters=5,
            classif_alg='RF',
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
            compute_errors=True
        )

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 11))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 11))
        self.assertEqual(errors.dtype, np.int16)

    def test_predict_by_machine_learner__LR_kNN_SAM_L8_S2(self):
        """Test linear regression using kNN_SAM from Landsat-8 to Sentinel-2A."""
        with self.assertRaises(NotImplementedError):
            self.SpH.predict_by_machine_learner(
                self.testArr_L8,
                method='LR', n_clusters=50,
                classif_alg='kNN_SAM',
                src_satellite='Landsat-8', src_sensor='OLI_TIRS',
                src_LBA=['1', '2', '3', '4', '5', '6', '7'],
                tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
                tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
                compute_errors=True
            )

    @unittest.SkipTest  # FIXME: RR classifiers are missing in resources
    def test_predict_by_machine_learner__RR_L8_S2(self):
        """Test ridge regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='RR', n_clusters=1,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
            compute_errors=True)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 11))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 11))
        self.assertEqual(errors.dtype, np.int16)

    def test_predict_by_machine_learner__QR_L8_S2(self):
        """Test quadratic regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='QR', n_clusters=1,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
            compute_errors=True)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 11))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 11))
        self.assertEqual(errors.dtype, np.int16)

    def test_predict_by_machine_learner__QR_cluster_L8_S2(self):
        """Test quadratic regression including spectral clusters from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='QR', n_clusters=50,
            classif_alg='MinDist',
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            # src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
            compute_errors=True,
            # compute_errors=False,
            src_nodataVal=-9999)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 11))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 11))
        self.assertEqual(errors.dtype, np.int16)

    @unittest.SkipTest  # FIXME RFR classifiers are missing (cannot be added to the repository to to file size > 1 GB)
    def test_predict_by_machine_learner__RFR_L8_S2(self):
        """Test random forest regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='RFR', n_clusters=1,
            classif_alg='MinDist',
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            # src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
            compute_errors=True,
            # compute_errors=False,
            src_nodataVal=-9999)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 11))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 11))
        self.assertEqual(errors.dtype, np.int16)


class Test_RSImage_ClusterPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.n_clusters = 50
        self.CP_SAMcla = RSImage_ClusterPredictor(method='LR', n_clusters=self.n_clusters,
                                                  classif_alg='SAM',
                                                  classifier_rootDir=os.path.join(testdata_rootdir, 'classifiers',
                                                                                  'SAMclassassignment'),
                                                  # **dict(n_neighbors=5)  # only compatible with kNN classif algs
                                                  )
        self.clf_L8 = self.CP_SAMcla.get_classifier(src_satellite='Landsat-8', src_sensor='OLI_TIRS',
                                                    src_LBA=['1', '2', '3', '4', '5', '6', '7'],
                                                    tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
                                                    tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'])

    def test_predict(self):
        for clf_alg in ['kNN_SAM', 'kNN_FEDSA', 'kNN_MinDist', 'MinDist', 'SAM', 'SID']:
            self.CP_SAMcla.classif_alg = clf_alg
            self.CP_SAMcla.classif_map = None  # reset classification map

            # build a testimage consisting of the cluster centers
            im_src = spectra2im(self.clf_L8.cluster_centers, 1, self.n_clusters)

            # predict
            im_homo = self.CP_SAMcla.predict(im_src, self.clf_L8, global_clf_threshold=4)

            # classifier should predict almost the target sensor center spectra
            if clf_alg not in ['kNN_MinDist', 'kNN_SAM', 'kNN_FEDSA']:
                self.assertTrue(np.array_equal(self.CP_SAMcla.classif_map[:].flatten(), np.arange(self.n_clusters)))
            self.assertTrue(np.allclose(im_homo, np.vstack([self.clf_L8.MLdict[i].tgt_cluster_center
                                                            for i in range(self.n_clusters)]),
                                        atol=5 if clf_alg not in ['kNN_MinDist', 'kNN_SAM', 'kNN_FEDSA'] else 1000))
