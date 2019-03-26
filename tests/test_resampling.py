#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_resampling
---------------

Tests for spechomo.resampling
"""

import unittest
import numpy as np
import os

from geoarray import GeoArray

from spechomo import __path__
from gms_preprocessing.io.input_reader import SRF  # FIXME
from spechomo.resampling import SpectralResampler as SR
from gms_preprocessing.model.gms_object import GMS_identifier  # FIXME


testdata = os.path.join(__path__[0], '../tests/data/Bavaria_farmland_LMU_Hyspex_subset.bsq')


class Test_SpectralResampler(unittest.TestCase):
    """Tests class for spechomo.resampling.SpectralResampler"""

    @classmethod
    def setUpClass(cls):
        cls.geoArr = GeoArray(testdata)
        cls.geoArr.to_mem()

        # get SRF instance for Landsat-8
        cls.srf_l8 = SRF(GMS_identifier(satellite='Landsat-8', sensor='OLI_TIRS', subsystem='',
                                        dataset_ID=-9999, logger=None, image_type='RSD', proc_level='L1A'))

    def test_resample_signature(self):
        # Get a hyperspectral spectrum.
        spectrum_wvl = np.array(self.geoArr.meta.band_meta['wavelength'], dtype=np.float).flatten()
        spectrum = self.geoArr[0, 0, :].flatten()

        sr = SR(spectrum_wvl, self.srf_l8)
        sig_rsp = sr.resample_signature(spectrum)
        self.assertTrue(np.any(sig_rsp), msg='Output signature is empty.')

    def test_resample_image(self):
        # Get a hyperspectral spectrum.
        image_wvl = np.array(self.geoArr.meta.band_meta['wavelength'], dtype=np.float).flatten()
        image = self.geoArr[:]

        sr = SR(image_wvl, self.srf_l8)
        im_rsp = sr.resample_image(image)
        self.assertTrue(np.any(im_rsp), msg='Output image is empty.')
