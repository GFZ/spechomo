# -*- coding: utf-8 -*-

from multiprocessing import Pool
from typing import Union  # noqa F401  # flake8 issue

import numpy as np
import scipy as sp
from geoarray import GeoArray
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from .logging import SpecHomo_Logger

# dependencies to get rid of
from gms_preprocessing.io.input_reader import SRF  # noqa F401  # flake8 issue


class SpectralResampler(object):
    """Class for spectral resampling of a single spectral signature (1D-array) or an image (3D-array)."""

    def __init__(self, wvl_src, srf_tgt, logger=None):
        # type: (np.ndarray, SRF, str) -> None
        """Get an instance of the SpectralResampler class.

        :param wvl_src:     center wavelength positions of the source spectrum
        :param srf_tgt:     spectral response of the target instrument as an instance of io.Input_reader.SRF.
        """
        # privates
        self._wvl_1nm = None
        self._srf_1nm = {}

        wvl_src = np.array(wvl_src, dtype=np.float).flatten()
        if srf_tgt.wvl_unit != 'nanometers':
            srf_tgt.convert_wvl_unit()

        self.wvl_src_nm = wvl_src if max(wvl_src) > 100 else wvl_src * 1000
        self.srf_tgt = srf_tgt
        self.logger = logger or SpecHomo_Logger(__name__)  # must be pickable

    @property
    def wvl_1nm(self):
        # spectral resampling of input image to 1 nm resolution
        if self._wvl_1nm is None:
            self._wvl_1nm = np.arange(np.ceil(self.wvl_src_nm.min()), np.floor(self.wvl_src_nm.max()), 1)
        return self._wvl_1nm

    @property
    def srf_1nm(self):
        if not self._srf_1nm:
            for band in self.srf_tgt.bands:
                # resample srf to 1 nm
                self._srf_1nm[band] = \
                    sp.interpolate.interp1d(self.srf_tgt.srfs_wvl, self.srf_tgt.srfs[band],
                                            bounds_error=False, fill_value=0, kind='linear')(self.wvl_1nm)

                # validate
                assert len(self._srf_1nm[band]) == len(self.wvl_1nm)

        return self._srf_1nm

    def resample_signature(self, spectrum, scale_factor=10000, nodataVal=None, v=False):
        # type: (np.ndarray, int, Union[int, float], bool) -> np.ndarray
        """Resample the given spectrum according to the spectral response functions of the target instument.

        :param spectrum:        spectral signature data
        :param scale_factor:    the scale factor to apply to the given spectrum when it is plotted (default: 10000)
        :param nodataVal:       no data value to be respected during resampling
        :param v:               enable verbose mode (shows a plot of the resampled spectrum) (default: False)
        :return:    resampled spectral signature
        """
        if not spectrum.ndim == 1:
            raise ValueError("The array of the given spectral signature must be 1-dimensional. "
                             "Received a %s-dimensional array." % spectrum.ndim)
        spectrum = np.array(spectrum, dtype=np.float).flatten()
        assert spectrum.size == self.wvl_src_nm.size

        # resample input spectrum and wavelength to 1nm
        if nodataVal is not None:
            spectrum = spectrum.astype(np.float)
            spectrum[spectrum == nodataVal] = np.nan

        spectrum_1nm = interp1d(self.wvl_src_nm, spectrum,
                                bounds_error=False, fill_value=0, kind='linear')(self.wvl_1nm)

        if v:
            plt.figure()
            plt.plot(self.wvl_1nm, spectrum_1nm/scale_factor, '.')

        if nodataVal is not None:
            spectrum_1nm = np.ma.masked_invalid(spectrum_1nm, nodataVal)

        spectrum_rsp = []

        for band, wvl_center in zip(self.srf_tgt.bands, self.srf_tgt.wvl):
            # compute the resampled spectral value (np.average computes the weighted mean value)
            if nodataVal is not None:
                specval_rsp = np.ma.average(spectrum_1nm, weights=self.srf_1nm[band])
                if not specval_rsp and specval_rsp != 0:
                    specval_rsp = nodataVal

            else:
                specval_rsp = np.ma.average(spectrum_1nm, weights=self.srf_1nm[band])

            if v:
                plt.plot(self.wvl_1nm, self.srf_1nm[band]/max(self.srf_1nm[band]))
                plt.plot(wvl_center, specval_rsp/scale_factor, 'x', color='r')

            spectrum_rsp.append(specval_rsp)

        # FIXME spectrum_rsp still contains NaNs if nodataVal is given
        return np.array(spectrum_rsp)

    def resample_spectra(self, spectra, chunksize=200, CPUs=None):
        # type: (Union[GeoArray, np.ndarray], int, Union[None, int]) -> np.ndarray
        """Resample the given spectral signatures according to the spectral response functions of the target instrument.

        :param spectra:     spectral signatures, provided as 2D array
                            (rows: spectral samples, columns: spectral information / bands)
        :param chunksize:   defines how many spectral signatures are resampled per CPU
        :param CPUs:        CPUs to use for processing
        """
        # input validation
        if not spectra.ndim == 2:
            ValueError("The given spectra array must be 2-dimensional. Received a %s-dimensional array."
                       % spectra.ndim)
        assert spectra.shape[1] == self.wvl_src_nm.size

        # convert spectra to one multispectral image column
        im_col = spectra.reshape(spectra.shape[0], 1, spectra.shape[1])

        im_col_rsp = self.resample_image(im_col, tiledims=(1, chunksize), CPUs=CPUs)
        spectra_rsp = im_col_rsp.reshape(im_col_rsp.shape[0], im_col_rsp.shape[2])

        return spectra_rsp

    def resample_image(self, image_cube, tiledims=(20, 20), CPUs=None):
        # type: (Union[GeoArray, np.ndarray], tuple, Union[None, int]) -> np.ndarray
        """Resample the given spectral image cube according to the spectral response functions of the target instrument.

        :param image_cube:      image (3D array) containing the spectral information in the third dimension
        :param tiledims:        dimension of tiles to be used during computation (rows, columns)
        :param CPUs:            CPUs to use for processing
        :return:    resampled spectral image cube
        """
        # input validation
        if not image_cube.ndim == 3:
            raise ValueError("The given image cube must be 3-dimensional. Received a %s-dimensional array."
                             % image_cube.ndim)
        assert image_cube.shape[2] == self.wvl_src_nm.size

        image_cube = GeoArray(image_cube)

        (R, C), B = image_cube.shape[:2], len(self.srf_tgt.bands)
        image_rsp = np.zeros((R, C, B), dtype=image_cube.dtype)

        if CPUs is None or CPUs > 1:
            with Pool(CPUs) as pool:
                tiles_rsp = pool.starmap(self._specresample, image_cube.tiles(tiledims))

        else:
            tiles_rsp = [self._specresample(bounds, tiledata) for bounds, tiledata in image_cube.tiles(tiledims)]

        for ((rS, rE), (cS, cE)), tile_rsp in tiles_rsp:
            image_rsp[rS: rE + 1, cS: cE + 1, :] = tile_rsp

        return image_rsp

    def _specresample(self, tilebounds, tiledata):
        # spectral resampling of input image to 1 nm resolution
        tile_1nm = interp1d(self.wvl_src_nm, tiledata,
                            axis=2, bounds_error=False, fill_value=0, kind='linear')(self.wvl_1nm)

        tile_rsp = np.zeros((*tile_1nm.shape[:2], len(self.srf_tgt.bands)), dtype=tiledata.dtype)
        for band_idx, band in enumerate(self.srf_tgt.bands):
            # compute the resampled image cube (np.average computes the weighted mean value)
            res = np.average(tile_1nm, weights=self.srf_1nm[band], axis=2)
            # NOTE: rounding here is important to prevent -9999. converted to -9998
            tile_rsp[:, :, band_idx] = res if np.issubdtype(tile_rsp.dtype, np.floating) else np.around(res)

        return tilebounds, tile_rsp
