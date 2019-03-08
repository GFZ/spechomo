#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_prediction
---------------

Tests for spechomo.prediction
"""


import unittest
import os
import numpy as np
from geoarray import GeoArray

from spechomo.prediction import SpectralHomogenizer
from spechomo import __path__ as spechomo_rootdir

classifier_rootdir = os.path.join(spechomo_rootdir, 'resources', 'classifiers')


class Test_SpectralHomogenizer(unittest.TestCase):
    """Tests class for spechomo.SpectralHomogenizer"""

    @classmethod
    def setUpClass(cls):
        cls.SpH = SpectralHomogenizer(classifier_rootDir=classifier_rootdir)
        cls.testArr_L8 = GeoArray(np.random.randint(1, 10000, (50, 50, 7), dtype=np.int16))  # no band 9, no pan
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
            method='LR', n_clusters=1,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '9', '10', '11', '12'],
            compute_errors=True
        )

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 13))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 13))
        self.assertEqual(errors.dtype, np.int16)

    def test_predict_by_machine_learner__RR_L8_S2(self):
        """Test ridge regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='RR', n_clusters=1,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '9', '10', '11', '12'],
            compute_errors=True)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 13))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 13))
        self.assertEqual(errors.dtype, np.int16)

    def test_predict_by_machine_learner__QR_L8_S2(self):
        """Test quadratic regression from Landsat-8 to Sentinel-2A."""
        predarr, errors = self.SpH.predict_by_machine_learner(
            self.testArr_L8,
            method='QR', n_clusters=1,
            src_satellite='Landsat-8', src_sensor='OLI_TIRS',
            src_LBA=['1', '2', '3', '4', '5', '6', '7'],
            tgt_satellite='Sentinel-2A', tgt_sensor='MSI',
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '9', '10', '11', '12'],
            compute_errors=True)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 13))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 13))
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
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '9', '10', '11', '12'],
            compute_errors=True,
            # compute_errors=False,
            nodataVal=-9999)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 13))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 13))
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
            tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '9', '10', '11', '12'],
            compute_errors=True,
            # compute_errors=False,
            nodataVal=-9999)

        self.assertIsInstance(predarr, GeoArray)
        self.assertEqual(predarr.shape, (50, 50, 13))
        self.assertEqual(predarr.dtype, np.int16)

        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, (50, 50, 13))
        self.assertEqual(errors.dtype, np.int16)
