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
test_resampling
---------------

Tests for spechomo.resampling
"""

import unittest
import numpy as np
import os

from geoarray import GeoArray
from pyrsr import RSR

from spechomo import __path__
from spechomo.resampling import SpectralResampler as SR
# noinspection PyProtectedMember
from spechomo.resampling import _resample_tile_mp, _initializer_mp


testdata = os.path.join(__path__[0], '../tests/data/Bavaria_farmland_LMU_Hyspex_subset.bsq')


class Test_SpectralResampler(unittest.TestCase):
    """Tests class for spechomo.resampling.SpectralResampler"""

    @classmethod
    def setUpClass(cls):
        cls.geoArr = GeoArray(testdata)
        cls.geoArr.to_mem()

        # get RSR instance for Landsat-8
        cls.rsr_l8 = RSR(satellite='Landsat-8', sensor='OLI_TIRS', no_thermal=True)

    def test_resample_signature(self):
        # Get a hyperspectral spectrum.
        spectrum_wvl = np.array(self.geoArr.meta.band_meta['wavelength'], dtype=np.float).flatten()
        spectrum = self.geoArr[0, 0, :].flatten()

        sr = SR(spectrum_wvl, self.rsr_l8)
        sig_rsp = sr.resample_signature(spectrum)
        self.assertTrue(np.any(sig_rsp), msg='Output signature is empty.')

    def test_resample_signature_with_nodata(self):
        # Get a hyperspectral spectrum.
        spectrum_wvl = np.array(self.geoArr.meta.band_meta['wavelength'], dtype=np.float).flatten()
        spectrum = self.geoArr[0, 0, :].flatten()
        spectrum[130: 140] = -9999

        sr = SR(spectrum_wvl, self.rsr_l8)
        sig_rsp = sr.resample_signature(spectrum, nodataVal=-9999)
        self.assertTrue(np.any(sig_rsp), msg='Output signature is empty.')

    def test_resample_image(self):
        # Get a hyperspectral image.
        image_wvl = np.array(self.geoArr.meta.band_meta['wavelength'], dtype=np.float).flatten()
        image = self.geoArr[:]

        sr = SR(image_wvl, self.rsr_l8)
        im_rsp = sr.resample_image(image)
        self.assertTrue(np.any(im_rsp), msg='Output image is empty.')

    def test__resample_tile_with_nodata(self):
        # Get a hyperspectral image tile.
        image_wvl = np.array(self.geoArr.meta.band_meta['wavelength'], dtype=np.float).flatten()
        tile = self.geoArr[:10, :5, :]
        tile[0, 0, 1] = -9999  # pixel with one band nodata
        tile[0, 1, 2] = -9999  # pixel with one band nodata
        tile[0, 2, 3] = -9999  # pixel with one band nodata
        tile[0, 3, :4] = -9999  # pixel with one band nodata
        tile[0, 4, 7] = -9999  # pixel with one band nodata
        tile[1:3, 1:3, :] = -9999  # pixels with all bands nodata

        sr = SR(image_wvl, self.rsr_l8)
        _initializer_mp(tile, sr.rsr_tgt, sr.rsr_1nm, sr.wvl_src_nm, sr.wvl_1nm)
        tilebounds, tile_rsp = _resample_tile_mp(((0, 9), (0, 4)),
                                                 nodataVal=-9999, alg_nodata='radical')
        self.assertTrue(np.any(tile_rsp), msg='Output image is empty.')
        self.assertTrue(np.all(tile_rsp[0, :4, :2] == -9999))
        self.assertTrue(tile_rsp[0, 4, 1] == -9999)

        tilebounds, tile_rsp = _resample_tile_mp(((0, 9), (0, 4)),
                                                 nodataVal=-9999, alg_nodata='conservative')
        self.assertTrue(np.any(tile_rsp), msg='Output image is empty.')
        self.assertTrue(np.all(tile_rsp[0, 3, 0] == -9999))
