# -*- coding: utf-8 -*-

"""Main module."""

import os
import numpy as np
import logging
from scipy.interpolate import interp1d
from typing import Union, List, Tuple  # noqa F401  # flake8 issue
from multiprocessing import cpu_count
from collections import OrderedDict
import dill
from pprint import pformat
import traceback
import zipfile
import tempfile
import time

from geoarray import GeoArray  # noqa F401  # flake8 issue

from .exceptions import ClassifierNotAvailableError
from .classifier_creation import get_machine_learner, get_filename_classifier_collection
from .logging import SpecHomo_Logger
from . import __path__ as spechomo_rootdir

__author__ = 'Daniel Scheffler'

classifier_rootdir = os.path.join(spechomo_rootdir, 'resources', 'classifiers')


class SpectralHomogenizer(object):
    """Class for applying spectral homogenization by applying an interpolation or machine learning approach."""
    def __init__(self, classifier_rootDir='', logger=None):
        """Get instance of SpectralHomogenizer.

        :param classifier_rootDir:  root directory where machine learning classifiers are stored.
        :param logger:              instance of logging.Logger
        """
        self.classifier_rootDir = classifier_rootDir or classifier_rootdir
        self.logger = logger or SpecHomo_Logger(__name__)

    def interpolate_cube(self, arrcube, source_CWLs, target_CWLs, kind='linear'):
        # type: (Union[np.ndarray, GeoArray], list, list, str) -> np.ndarray
        """Spectrally interpolate the spectral bands of a remote sensing image to new band positions.

        :param arrcube:     array to be spectrally interpolated
        :param source_CWLs: list of source central wavelength positions
        :param target_CWLs: list of target central wavelength positions
        :param kind:        interpolation kind to be passed to scipy.interpolate.interp1d (default: 'linear')
        :return:
        """
        assert kind in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'], \
            "%s is not a supported kind of spectral interpolation." % kind
        assert arrcube is not None,\
            'L2B_obj.interpolate_cube_linear expects a numpy array as input. Got %s.' % type(arrcube)

        orig_CWLs, target_CWLs = np.array(source_CWLs), np.array(target_CWLs)

        self.logger.info(
            'Performing spectral homogenization (%s interpolation) with target wavelength positions at %s nm.'
            % (kind, ', '.join(np.array(target_CWLs[:-1]).astype(str)) + ' and %s' % target_CWLs[-1]))
        outarr = interp1d(np.array(orig_CWLs), arrcube, axis=2, kind=kind, fill_value='extrapolate')(target_CWLs)
        outarr = outarr.astype(np.int16)

        assert outarr.shape == tuple([*arrcube.shape[:2], len(target_CWLs)])

        return outarr

    def predict_by_machine_learner(self, arrcube, method, src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor,
                                   tgt_LBA, n_clusters=50, classif_alg='MinDist', kNN_n_neighbors=10,
                                   nodataVal=None, compute_errors=False, bandwise_errors=True, **fallback_argskwargs):
        # type: (Union[np.ndarray, GeoArray], str, str, str, list, str, str, list, int, str, int, int, ...) -> tuple
        """Predict spectral bands of target sensor by applying a machine learning approach.

        :param arrcube:             input image array for target sensor spectral band prediction (rows x cols x bands)
        :param method:              machine learning approach to be used for spectral bands prediction
                                    'LR': Linear Regression
                                    'RR': Ridge Regression
                                    'QR': Quadratic Regression
                                    'RFR':  Random Forest Regression  (50 trees; does not allow spectral sub-clustering)
        :param src_satellite:       source satellite, e.g., 'Landsat-8'
        :param src_sensor:          source sensor, e.g., 'OLI_TIRS'
        :param src_LBA:             source LayerBandsAssignment
        :param tgt_satellite:       target satellite, e.g., 'Landsat-8'
        :param tgt_sensor:          target sensor, e.g., 'OLI_TIRS'
        :param tgt_LBA:             target LayerBandsAssignment
        :param n_clusters:          Number of spectral clusters to be used during LR/ RR/ QR homogenization.
                                    E.g., 50 means that the image to be converted to the spectral target sensor
                                    is clustered into 50 spectral clusters and one separate machine learner per
                                    cluster is applied to the input data to predict the homogenized image. If
                                    'spechomo_n_clusters' is set to 1, the source image is not clustered and
                                    only one machine learning classifier is used for prediction.
        :param classif_alg:         Multispectral classification algorithm to be used to determine the spectral cluster
                                    each pixel belongs to.
                                    'MinDist': Minimum Distance (Nearest Centroid) Classification
                                    'kNN': k-Nearest-Neighbor Classification
                                    'SAM': Spectral Angle Mapping
                                    'SID': spectral information divergence
        :param kNN_n_neighbors:     The number of neighbors to be considered in case 'classif_alg' is set to 'kNN'.
                                    Otherwise, this parameter is ignored.
        :param nodataVal:           no data value
        :param compute_errors:      whether to compute pixel- / bandwise model errors for estimated pixel values
                                    (default: false)
        :param bandwise_errors      whether to compute error information for each band separately (True - default)
                                    or to average errors over bands using median (False) (ignored in case of fallback)
        :param fallback_argskwargs: arguments and keyword arguments for fallback algorithm ({'args':{}, 'kwargs': {}}
        :return:                    predicted array (rows x columns x bands)
        :rtype:                     Tuple[np.ndarray, Union[np.ndarray, None]]
        """
        # TODO: add LBA validation to .predict()
        kw = dict(method=method,
                  classifier_rootDir=self.classifier_rootDir,
                  n_clusters=n_clusters,
                  classif_alg=classif_alg)

        if classif_alg == 'kNN':
            kw['n_neighbors'] = kNN_n_neighbors

        PR = RSImage_ClusterPredictor(**kw)

        ######################
        # get the classifier #
        ######################

        cls = None
        exc = Exception()
        try:
            cls = PR.get_classifier(src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA)

        except FileNotFoundError as e:
            self.logger.warning('No machine learning classifier available that fulfills the specifications of the '
                                'spectral reference sensor. Falling back to linear interpolation for performing '
                                'spectral homogenization.')
            exc = e

        except ClassifierNotAvailableError as e:
            self.logger.error('\nAn error occurred during spectral homogenization using machine learning. '
                              'Falling back to linear interpolation. Error message was: ')
            self.logger.error(traceback.format_exc())
            exc = e

        ##################
        # run prediction #
        ##################

        errors = None
        if cls:
            self.logger.info('Performing spectral homogenization using %s. Target is %s %s %s.'
                             % (method, tgt_satellite, tgt_sensor, tgt_LBA))
            im_homo = PR.predict(arrcube, classifier=cls, in_nodataVal=nodataVal, cmap_nodataVal=nodataVal)
            if compute_errors:
                errors = PR.compute_prediction_errors(im_homo, cls, nodataVal=nodataVal, cmap_nodataVal=nodataVal)

                if not bandwise_errors:
                    errors = np.median(errors, axis=2).astype(errors.dtype)

        elif fallback_argskwargs:
            # fallback: use linear interpolation and set errors to an array of zeros
            im_homo = self.interpolate_cube(arrcube, *fallback_argskwargs['args'], **fallback_argskwargs['kwargs'])
            if compute_errors:
                self.logger.warning("Spectral homogenization algorithm had to be performed by linear interpolation "
                                    "(fallback). Unable to compute any accuracy information from that.")
                if not bandwise_errors:
                    errors = np.zeros_like(im_homo, dtype=np.int16)
                else:
                    errors = np.zeros(im_homo.shape[:2], dtype=np.int16)

        else:
            raise exc

        return im_homo, errors


def im2spectra(geoArr):
    # type: (Union[GeoArray, np.ndarray]) -> np.ndarray
    """Convert 3D images to array of spectra samples (rows: samples;  cols: spectral information)."""
    return geoArr.reshape((geoArr.shape[0] * geoArr.shape[1], geoArr.shape[2]))


def spectra2im(spectra, tgt_rows, tgt_cols):
    # type: (Union[GeoArray, np.ndarray], int, int) -> np.ndarray
    """Convert array of spectra samples (rows: samples;  cols: spectral information) to a 3D image.

    :param spectra:     2D array with rows: spectral samples / columns: spectral information (bands)
    :param tgt_rows:    number of target image rows
    :param tgt_cols:    number of target image rows
    :return:            3D array (rows x columns x spectral bands)
    """
    return spectra.reshape(tgt_rows, tgt_cols, spectra.shape[1])


class ClassifierCollection(object):
    def __init__(self, path_dillFile):
        with open(path_dillFile, 'rb') as inF:
            self.content = dill.load(inF)

    def __repr__(self):
        """Returns representation of ClassifierCollection.

        :return: e.g., "{'1__2__3__4__5__7': {('Landsat-5', 'TM'): {'1__2__3__4__5__7':
                        LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)}, ..."
        """
        return pformat(self.content)

    def __getitem__(self, item):
        return self.content[item]


class RSImage_ClusterPredictor(object):
    """Predictor class applying the predict() function of a machine learning classifier described by the given args."""
    def __init__(self, method='LR', n_clusters=50, classif_alg='MinDist', classifier_rootDir='',
                 CPUs=1, logger=None, **kw_clf_init):
        # type: (str, int, str, str, Union[None, int], logging.Logger, dict) -> None
        """Get an instance of RSImage_ClusterPredictor.

        :param method:              machine learning approach to be used for spectral bands prediction
                                    'LR':   Linear Regression
                                    'RR':   Ridge Regression
                                    'QR':   Quadratic Regression
                                    'RFR':  Random Forest Regression  (50 trees; does not allow spectral sub-clustering)
        :param n_clusters:          Number of spectral clusters to be used during LR/ RR/ QR homogenization.
                                    E.g., 50 means that the image to be converted to the spectral target sensor
                                    is clustered into 50 spectral clusters and one separate machine learner per
                                    cluster is applied to the input data to predict the homogenized image. If
                                    'n_clusters' is set to 1, the source image is not clustered and
                                    only one machine learning classifier is used for prediction.
        :param classif_alg:         algorithm to be used for image classification
                                    (to define which cluster each pixel belongs to)
                                    'MinDist': Minimum Distance (Nearest Centroid)
                                    'kNN': k-nearest-neighbour
                                    'SAM': spectral angle mapping
                                    'SID': spectral information divergence
        :param classifier_rootDir:  root directory where machine learning classifiers are stored.
        :param CPUs:                number of CPUs to use
        :param logger:              instance of logging.Logger()
        :param kw_clf_init          keyword arguments to be passed to classifier init functions if possible
        """
        self.method = method
        self.n_clusters = n_clusters
        self.classifier_rootDir = os.path.abspath(classifier_rootDir)
        self.classif_map = None
        self.CPUs = CPUs
        self.classif_alg = classif_alg
        self.logger = logger or SpecHomo_Logger(__name__)  # must be pickable
        self.kw_clf_init = kw_clf_init

        # validate
        if method == 'RFR' and n_clusters > 1:
            self.logger.warning("The spectral homogenization method 'Random Forest Regression' does not allow spectral "
                                "sub-clustering. Setting 'n_clusters' to 1.")
            self.n_clusters = 1

        if self.classif_alg == 'kNN' and 'n_neighbors' in kw_clf_init and self.n_clusters < kw_clf_init['n_neighbors']:
            self.kw_clf_init['n_neighbors'] = self.n_clusters

    def get_classifier(self, src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA):
        # type: (str, str, list, str, str, list) -> Cluster_Learner
        """Select the correct machine learning classifier out of previously saves classifier collections.

        Describe the classifier specifications with the given arguments.
        :param src_satellite:   source satellite, e.g., 'Landsat-8'
        :param src_sensor:      source sensor, e.g., 'OLI_TIRS'
        :param src_LBA:         source LayerBandsAssignment
        :param tgt_satellite:   target satellite, e.g., 'Landsat-8'
        :param tgt_sensor:      target sensor, e.g., 'OLI_TIRS'
        :param tgt_LBA:         target LayerBandsAssignment
        :return:                classifier instance loaded from disk
        """
        # get path of classifier zip archive
        path_classifier_zip = os.path.join(self.classifier_rootDir, '%s_classifiers.zip' % self.method)
        if not os.path.isfile(path_classifier_zip):
            raise FileNotFoundError("No '%s' classifiers available at %s." % (self.method, path_classifier_zip))

        # create an instance of ClusterLearner by reading the requested classifier from the zip archive
        with zipfile.ZipFile(path_classifier_zip, "r") as zf, tempfile.TemporaryDirectory() as td:
            # read requested classifier from zip archive and create a ClassifierCollection
            fName_cls = \
                get_filename_classifier_collection(self.method, src_satellite, src_sensor, n_clusters=self.n_clusters)

            try:
                zf.extract(fName_cls, td)
                path_cls = os.path.join(td, fName_cls)
                dict_clust_MLinstances = \
                    ClassifierCollection(path_cls)['__'.join(src_LBA)][tgt_satellite, tgt_sensor]['__'.join(tgt_LBA)]
            except KeyError:
                raise ClassifierNotAvailableError(self.method, src_satellite, src_sensor, src_LBA,
                                                  tgt_satellite, tgt_sensor, tgt_LBA, self.n_clusters)

            # validation
            expected_MLtype = type(get_machine_learner(self.method))
            for label, ml in dict_clust_MLinstances.items():
                if not isinstance(ml, expected_MLtype):
                    raise ValueError("The given dillFile %s contains a spectral cluster (label '%s') with a %s machine "
                                     "learner instead of the expected %s."
                                     % (os.path.basename(fName_cls), label, type(ml), expected_MLtype.__name__,))

            return Cluster_Learner(dict_clust_MLinstances)

    def predict(self, image, classifier, in_nodataVal=None, out_nodataVal=None, cmap_nodataVal=None, CPUs=1):
        # type: (Union[np.ndarray, GeoArray], Cluster_Learner, float, float, float, int) -> GeoArray
        """Apply the prediction function of the given specifier to the given remote sensing image.

        # NOTE: The 'nodataVal' is written

        :param image:           3D array representing the input image
        :param classifier:      the classifier instance
        :param in_nodataVal:    no data value of the input image
                                (auto-computed if not given or contained in image GeoArray)
        :param out_nodataVal:   no data value written into the predicted image
                                (copied from the input image if not given)
        :param cmap_nodataVal:  no data value for the classification map
                                in case more than one sub-classes are used for prediction
        :param CPUs:            CPUs to use (default: 1)
        :return:                3D array representing the predicted spectral image cube
        """
        image = image if isinstance(image, GeoArray) else GeoArray(image, nodata=in_nodataVal)

        # ensure image.nodata is present (important for classify_image() -> overwrites cmap at nodata positions)
        image.nodata = in_nodataVal if in_nodataVal is not None else image.nodata  # might be auto-computed here

        # assign each input pixel to a cluster (compute classification with cluster centers as endmembers)
        if not self.classif_map:
            if self.n_clusters > 1:
                t0 = time.time()
                kw_clf = dict(classif_alg=self.classif_alg,
                              in_nodataVal=image.nodata,
                              cmap_nodataVal=cmap_nodataVal,  # written into classif_map at nodata
                              CPUs=self.CPUs,
                              **self.kw_clf_init)

                self.classif_map = classify_image(image, classifier.cluster_centers, classifier.cluster_pixVals,
                                                  **kw_clf)

                self.logger.info('Total classification time: %s'
                                 % time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))

            else:
                self.classif_map = np.full((image.rows, image.cols), classifier.cluster_pixVals[0], np.int8)

        # adjust classifier
        if CPUs is None or CPUs > 1:
            # FIXME does not work -> parallelize with https://github.com/ajtulloch/sklearn-compiledtrees?
            classifier.n_jobs = cpu_count() if CPUs is None else CPUs

        # apply prediction
        # NOTE: prediction is applied in 1000 x 1000 tiles to save memory (because classifier.predict returns float32)
        out_nodataVal = out_nodataVal if out_nodataVal is not None else image.nodata
        image_predicted = GeoArray(np.empty((image.rows, image.cols, classifier.tgt_n_bands), dtype=image.dtype),
                                   geotransform=image.gt, projection=image.prj, nodata=out_nodataVal,
                                   bandnames=GMS_object.LBA2bandnames(classifier.tgt_LBA))

        t0 = time.time()
        for ((rS, rE), (cS, cE)), im_tile in image.tiles(tilesize=(1000, 1000)):
            self.logger.info('Predicting tile ((%s, %s), (%s, %s))...' % (rS, rE, cS, cE))

            classif_map_tile = self.classif_map[rS: rE+1, cS: cE+1]  # integer array

            # predict!
            im_tile_pred = \
                classifier.predict(im_tile, classif_map_tile,
                                   nodataVal=out_nodataVal, cmap_nodataVal=cmap_nodataVal).astype(image.dtype)
            image_predicted[rS:rE + 1, cS:cE + 1] = im_tile_pred

        self.logger.info('Total prediction time: %s' % time.strftime("%H:%M:%S", time.gmtime(time.time()-t0)))

        # re-apply nodata values to predicted result
        if image.nodata is not None:
            image_predicted[image.mask_nodata[:] == 0] = out_nodataVal

        # copy mask_nodata
        image_predicted.mask_nodata = image.mask_nodata

        # image_predicted.save(
        #     '/home/gfz-fe/scheffler/temp/SPECHOM_py/image_predicted_QRclust1_MinDist_noB9.bsq')
        # GeoArray(self.classif_map).save(
        #     '/home/gfz-fe/scheffler/temp/SPECHOM_py/classif_map_QRclust1_MinDist_noB9.bsq')

        return image_predicted

    def compute_prediction_errors(self, im_predicted, cluster_classifier, nodataVal=None, cmap_nodataVal=None):
        # type: (Union[np.ndarray, GeoArray], Cluster_Learner, float, float) -> np.ndarray
        """Compute errors that quantify prediction inaccurracy per band and per pixel.

        :param im_predicted:        3D array representing the predicted image
        :param cluster_classifier:  instance of Cluster_Learner
        :param nodataVal:           no data value of the input image
                                    (auto-computed if not given or contained in im_predicted GeoArray)
                                    NOTE: The value is also used as output nodata value for the errors array.
        :param cmap_nodataVal:      no data value for the classification map
                                    in case more than one sub-classes are used for prediction
        :return:                    3D array (int16) representing prediction errors per band and pixel
        """
        im_predicted = im_predicted if isinstance(im_predicted, GeoArray) else GeoArray(im_predicted, nodata=nodataVal)
        im_predicted.nodata = nodataVal if nodataVal is not None else im_predicted.nodata  # might be auto-computed here

        for cls in cluster_classifier:
            if not len(cls.rmse_per_band) == GeoArray(im_predicted).bands:
                raise ValueError('The given classifier contains error statistics incompatible to the shape of the '
                                 'image.')
        if self.classif_map is None:
            raise RuntimeError('self.classif_map must be generated by running self.predict() beforehand.')

        errors = np.empty_like(im_predicted)

        # iterate over all cluster labels and copy rmse values
        for pixVal in sorted(list(np.unique(self.classif_map))):
            if pixVal == cmap_nodataVal:
                continue

            self.logger.info('Inpainting error values for cluster #%s...' % pixVal)

            rmse_per_band_int = np.round(cluster_classifier.MLdict[pixVal].rmse_per_band, 0).astype(np.int16)
            errors[self.classif_map == pixVal] = rmse_per_band_int

        # TODO validate this equation
        # errors = (errors * im_predicted[:] / 10000).astype(errors.dtype)

        # re-apply nodata values to predicted result
        if im_predicted.nodata is not None:
            # errors[im_predicted == im_predicted.nodata] = im_predicted.nodata
            errors[im_predicted.mask_nodata.astype(np.int8) == 0] = im_predicted.nodata

        # GeoArray(errors).save('/home/gfz-fe/scheffler/temp/SPECHOM_py/errors_LRclust1_MinDist_noB9_clusterpred.bsq')

        return errors


class Cluster_Learner(object):
    def __init__(self, dict_clust_MLinstances):
        # type: (dict) -> None
        """

        :param dict_clust_MLinstances:
        """
        self.cluster_pixVals = list(sorted(dict_clust_MLinstances.keys()))  # type: List[int]
        self.MLdict = OrderedDict((clust, dict_clust_MLinstances[clust]) for clust in self.cluster_pixVals)
        sample_MLinst = list(self.MLdict.values())[0]
        self.src_satellite = sample_MLinst.src_satellite
        self.src_sensor = sample_MLinst.src_sensor
        self.tgt_satellite = sample_MLinst.tgt_satellite
        self.tgt_sensor = sample_MLinst.tgt_sensor
        self.src_LBA = sample_MLinst.src_LBA
        self.tgt_LBA = sample_MLinst.tgt_LBA
        self.src_n_bands = sample_MLinst.src_n_bands
        self.tgt_n_bands = sample_MLinst.tgt_n_bands
        self.src_wavelengths = sample_MLinst.src_wavelengths
        self.tgt_wavelengths = sample_MLinst.tgt_wavelengths
        self.n_clusters = sample_MLinst.n_clusters
        self.cluster_centers = np.array([cc.cluster_center for cc in self.MLdict.values()])

    def __iter__(self):
        for cluster in self.cluster_pixVals:
            yield self.MLdict[cluster]

    def predict(self, im_src, cmap, nodataVal=None, cmap_nodataVal=None):
        """

        :param im_src:
        :param cmap:            classification map that assigns each image spectrum to its corresponding cluster
                                -> must be a 1D np.ndarray with the same Y-dimension like src_spectra
        :param nodataVal:       nodata value to be used to fill into the predicted image
        :param cmap_nodataVal:  nodata class value of the nodata class of the classification map
        :return:
        """
        cluster_labels = sorted(list(np.unique(cmap)))

        im_pred = np.full((im_src.shape[0], im_src.shape[1], self.tgt_n_bands),
                          fill_value=nodataVal if nodataVal is not None else 0,
                          dtype=im_src.dtype)

        if len(cluster_labels) > 1:
            # iterate over all cluster labels and apply corresponding machine learner parameters
            # to predict target spectra

            for pixVal in cluster_labels:
                if pixVal == cmap_nodataVal:
                    continue

                classifier = self.MLdict[pixVal]
                mask_pixVal = cmap == pixVal
                im_pred[mask_pixVal] = classifier.predict(im_src[mask_pixVal]).astype(im_src.dtype)

        else:
            # predict target spectra directly (much faster than the above algorithm)
            pixVal = cluster_labels[0]

            if pixVal != cmap_nodataVal:
                spectra = im2spectra(im_src)
                classifier = self.MLdict[pixVal]
                spectra_pred = classifier.predict(spectra).astype(im_src.dtype)
                im_pred = spectra2im(spectra_pred, im_src.shape[0], im_src.shape[1])
            else:
                # im_src consists only of no data values
                pass  # im_pred keeps at nodataVal

        return im_pred
