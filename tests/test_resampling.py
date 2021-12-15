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


testdata = os.path.join(__path__[0], '../tests/data/AV_mastercal_testdata.bsq')


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

    @unittest.SkipTest
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


if __name__ == '__main__':
    import pytest
    pytest.main()
