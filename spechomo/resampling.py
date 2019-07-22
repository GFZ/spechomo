# -*- coding: utf-8 -*-

from multiprocessing import Pool
from typing import Union, TYPE_CHECKING  # noqa F401  # flake8 issue

import numpy as np
import scipy as sp
from geoarray import GeoArray
from geoarray.baseclasses import get_array_tilebounds
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from .logging import SpecHomo_Logger

# TODO dependencies to get rid of
if TYPE_CHECKING:
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

    def resample_signature(self, spectrum, scale_factor=10000, nodataVal=None, alg_nodata='radical', v=False):
        # type: (np.ndarray, int, Union[int, float], str, bool) -> np.ndarray
        """Resample the given spectrum according to the spectral response functions of the target instument.

        :param spectrum:        spectral signature data
        :param scale_factor:    the scale factor to apply to the given spectrum when it is plotted (default: 10000)
        :param nodataVal:       no data value to be respected during resampling
        :param alg_nodata:      algorithm how to deal with pixels where the spectral bands of the source image
                                contain nodata within the spectral response of a target band
                                'radical':      set output band to nodata
                                'conservative': use existing spectral information and ignore nodata
                                                (might alter the outpur spectral information,
                                                 e.g., at spectral absorption bands)
        :param v:               enable verbose mode (shows a plot of the resampled spectrum) (default: False)
        :return:    resampled spectral signature
        """
        if not spectrum.ndim == 1:
            raise ValueError("The array of the given spectral signature must be 1-dimensional. "
                             "Received a %s-dimensional array." % spectrum.ndim)
        spectrum = np.array(spectrum, dtype=np.float).flatten()
        assert spectrum.size == self.wvl_src_nm.size

        # convert spectrum to one multispectral image pixel and resample it
        spectrum_rsp = \
            self.resample_image(spectrum.reshape(1, 1, spectrum.size),
                                tiledims=(1, 1),
                                nodataVal=nodataVal,
                                alg_nodata=alg_nodata,
                                CPUs=1)\
            .ravel()

        if v:
            plt.figure()
            for band in self.srf_tgt.bands:
                plt.plot(self.wvl_1nm, self.srf_1nm[band] / max(self.srf_1nm[band]))
            plt.plot(self.wvl_src_nm, spectrum / scale_factor, '.')
            plt.plot(self.srf_tgt.wvl, spectrum_rsp / scale_factor, 'x', color='r')
            plt.show()

        return spectrum_rsp

    def resample_spectra(self, spectra, chunksize=200, nodataVal=None, alg_nodata='radical', CPUs=None):
        # type: (Union[GeoArray, np.ndarray], int, Union[int, float], str, Union[None, int]) -> np.ndarray
        """Resample the given spectral signatures according to the spectral response functions of the target instrument.

        :param spectra:     spectral signatures, provided as 2D array
                            (rows: spectral samples, columns: spectral information / bands)
        :param chunksize:   defines how many spectral signatures are resampled per CPU
        :param nodataVal:   no data value to be respected during resampling
        :param alg_nodata:  algorithm how to deal with pixels where the spectral bands of the source image
                            contain nodata within the spectral response of a target band
                            'radical':      set output band to nodata
                            'conservative': use existing spectral information and ignore nodata
                                                (might alter the outpur spectral information,
                                                 e.g., at spectral absorption bands)
        :param CPUs:        CPUs to use for processing
        """
        # input validation
        if not spectra.ndim == 2:
            ValueError("The given spectra array must be 2-dimensional. Received a %s-dimensional array."
                       % spectra.ndim)
        assert spectra.shape[1] == self.wvl_src_nm.size

        # convert spectra to one multispectral image column
        im_col = spectra.reshape(spectra.shape[0], 1, spectra.shape[1])

        im_col_rsp = self.resample_image(im_col,
                                         tiledims=(1, chunksize),
                                         nodataVal=nodataVal,
                                         alg_nodata=alg_nodata,
                                         CPUs=CPUs)
        spectra_rsp = im_col_rsp.reshape(im_col_rsp.shape[0], im_col_rsp.shape[2])

        return spectra_rsp

    def resample_image(self, image_cube, tiledims=(20, 20), nodataVal=None, alg_nodata='radical', CPUs=None):
        # type: (Union[GeoArray, np.ndarray], tuple, Union[int, float], str, Union[None, int]) -> np.ndarray
        """Resample the given spectral image cube according to the spectral response functions of the target instrument.

        :param image_cube:      image (3D array) containing the spectral information in the third dimension
        :param tiledims:        dimension of tiles to be used during computation (rows, columns)
        :param nodataVal:       nodata value of the input image
        :param alg_nodata:      algorithm how to deal with pixels where the spectral bands of the source image
                                contain nodata within the spectral response of a target band
                                'radical':      set output band to nodata
                                'conservative': use existing spectral information and ignore nodata
                                                (might alter the output spectral information,
                                                 e.g., at spectral absorption bands)
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

        initargs = image_cube, self.srf_tgt, self.srf_1nm, self.wvl_src_nm, self.wvl_1nm
        if CPUs is None or CPUs > 1:
            with Pool(CPUs, initializer=_initializer_mp, initargs=initargs) as pool:
                tiles_rsp = pool.starmap(_resample_tile_mp,
                                         [(bounds, nodataVal, alg_nodata)
                                          for bounds in get_array_tilebounds(image_cube.shape, tiledims)])

        else:
            _initializer_mp(*initargs)
            tiles_rsp = [_resample_tile_mp(bounds, nodataVal, alg_nodata)
                         for bounds in get_array_tilebounds(image_cube.shape, tiledims)]

        for ((rS, rE), (cS, cE)), tile_rsp in tiles_rsp:
            image_rsp[rS: rE + 1, cS: cE + 1, :] = tile_rsp

        return image_rsp


_globs_mp = dict()


def _initializer_mp(image_cube, srf_tgt, srf_1nm, wvl_src_nm, wvl_1nm):
    # type: (np.ndarray, SRF, SRF, np.ndarray, np.ndarray) -> None
    global _globs_mp
    _globs_mp.update(dict(
        image_cube=image_cube,
        srf_tgt=srf_tgt,
        srf_1nm=srf_1nm,
        wvl_src_nm=wvl_src_nm,
        wvl_1nm=wvl_1nm,
    ))


def _resample_tile_mp(tilebounds, nodataVal=None, alg_nodata='radical'):
    # TODO speed up by using numba as described here https://krstn.eu/fast-linear-1D-interpolation-with-numba/

    (rS, rE), (cS, cE) = tilebounds

    # get global share variables
    tiledata = _globs_mp['image_cube'][rS: rE + 1, cS: cE + 1, :]  # type: Union[GeoArray, np.ndarray]
    srf_tgt = _globs_mp['srf_tgt']
    srf_1nm = _globs_mp['srf_1nm']
    wvl_src_nm = _globs_mp['wvl_src_nm']
    wvl_1nm = _globs_mp['wvl_1nm']

    if alg_nodata not in ['radical', 'conservative']:
        raise ValueError(alg_nodata)

    tile_rsp = np.zeros((*tiledata.shape[:2], len(srf_tgt.bands)), dtype=tiledata.dtype)

    if nodataVal is not None:
        if np.isfinite(nodataVal):
            _mask_bool3d = tiledata == nodataVal
            mask_anynodata = np.any(_mask_bool3d, axis=2)
            mask_allnodata = np.all(_mask_bool3d, axis=2)
        else:
            raise NotImplementedError(nodataVal)

        tiledata = tiledata.astype(np.float)
        tiledata[_mask_bool3d] = np.nan

        # fill pixels with all bands nodata
        tile_rsp[mask_allnodata] = nodataVal

        # upsample spectra containing data to 1nm and get nan-mask
        tilespectra_1nm = interp1d(wvl_src_nm, tiledata[~mask_allnodata],
                                   axis=1, bounds_error=False, fill_value=0, kind='linear')(wvl_1nm)
        isnan = np.isnan(tilespectra_1nm)
        nan_in_spec = np.any(isnan, axis=1)

        # compute resampled values for pixels without nodata
        tilespectra_1nm_nonan = tilespectra_1nm[~nan_in_spec, :]
        for band_idx, band in enumerate(srf_tgt.bands):
            # compute the resampled image cube (np.average computes the weighted mean value)
            res = np.average(tilespectra_1nm_nonan, weights=srf_1nm[band], axis=1)
            # NOTE: rounding here is important to prevent -9999. converted to -9998
            if not np.issubdtype(tile_rsp.dtype, np.floating):
                res = np.around(res)

            tile_rsp[~mask_anynodata, band_idx] = res

        # compute resampled values for pixels with nodata
        tilespectra_1nm_withnan = tilespectra_1nm[nan_in_spec, :]
        tilespectra_1nm_withnan_ma = np.ma.masked_invalid(tilespectra_1nm_withnan)
        mask_withnan = mask_anynodata & ~mask_allnodata
        isnan_sub = isnan[nan_in_spec, :]

        for band_idx, band in enumerate(srf_tgt.bands):
            res_ma = np.ma.average(tilespectra_1nm_withnan_ma, axis=1, weights=srf_1nm[band])

            # in case all 1nm bands for the current band are NaN, the resulting average will also be NaN => fill
            res = np.ma.filled(res_ma, nodataVal)

            if alg_nodata == 'radical':
                # set those output values to nodata where the input bands within the target SRF contain any nodata
                badspec = np.any(isnan_sub & (srf_1nm[band] > 0), axis=1)
                res[badspec] = nodataVal

            if not np.issubdtype(tile_rsp.dtype, np.floating):
                res = np.around(res)

            tile_rsp[mask_withnan, band_idx] = res

    else:
        # spectral resampling of input image to 1 nm resolution
        tile_1nm = interp1d(wvl_src_nm, tiledata,
                            axis=2, bounds_error=False, fill_value=0, kind='linear')(wvl_1nm)

        for band_idx, band in enumerate(srf_tgt.bands):
            # compute the resampled image cube (np.average computes the weighted mean value)
            res = np.average(tile_1nm, weights=srf_1nm[band], axis=2)
            # NOTE: rounding here is important to prevent -9999. converted to -9998
            tile_rsp[:, :, band_idx] = res if np.issubdtype(tile_rsp.dtype, np.floating) else np.around(res)

    return tilebounds, tile_rsp
