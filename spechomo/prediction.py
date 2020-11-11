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

"""Main module."""

import os
import numpy as np
import logging  # noqa F401  # flake8 issue
from typing import Union, Tuple  # noqa F401  # flake8 issue
from multiprocessing import cpu_count
import traceback
import time
from geoarray import GeoArray  # noqa F401  # flake8 issue
from specclassify import classify_image
# from specclassify import kNN_MinimumDistance_Classifier

from .classifier import Cluster_Learner
from .exceptions import ClassifierNotAvailableError
from .logging import SpecHomo_Logger
from .options import options


__author__ = 'Daniel Scheffler'

_classifier_rootdir = options['classifiers']['rootdir']


class SpectralHomogenizer(object):
    """Class for applying spectral homogenization by applying an interpolation or machine learning approach."""

    def __init__(self, classifier_rootDir='', logger=None, CPUs=None):
        """Get instance of SpectralHomogenizer.

        :param classifier_rootDir:  root directory where machine learning classifiers are stored.
        :param logger:              instance of logging.Logger
        """
        self.classifier_rootDir = classifier_rootDir or _classifier_rootdir
        self.logger = logger or SpecHomo_Logger(__name__)
        self.CPUs = CPUs or cpu_count()

    def interpolate_cube(self, arrcube, source_CWLs, target_CWLs, kind='linear'):
        # type: (Union[np.ndarray, GeoArray], list, list, str) -> GeoArray
        """Spectrally interpolate the spectral bands of a remote sensing image to new band positions.

        :param arrcube:     array to be spectrally interpolated
        :param source_CWLs: list of source central wavelength positions
        :param target_CWLs: list of target central wavelength positions
        :param kind:        interpolation kind to be passed to scipy.interpolate.interp1d (default: 'linear')
        :return:
        """
        from scipy.interpolate import interp1d  # import here to avoid static TLS ImportError

        assert kind in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'], \
            "%s is not a supported kind of spectral interpolation." % kind
        assert arrcube is not None,\
            'L2B_obj.interpolate_cube_linear expects a numpy array as input. Got %s.' % type(arrcube)

        orig_CWLs = np.array(source_CWLs)
        target_CWLs = np.array(target_CWLs)

        self.logger.info(
            'Performing spectral homogenization (%s interpolation) with target wavelength positions at %s nm.'
            % (kind, ', '.join(np.round(np.array(target_CWLs[:-1]), 1).astype(str)) +
               ' and %s' % np.round(target_CWLs[-1], 1)))
        outarr = \
            interp1d(np.array(orig_CWLs),
                     arrcube,
                     axis=2,
                     kind=kind,
                     fill_value='extrapolate')(target_CWLs)

        if np.min(outarr) >= np.iinfo(np.int16).min and \
           np.max(outarr) <= np.iinfo(np.int16).max:

            outarr = outarr.astype(np.int16)

        elif np.min(outarr) >= np.iinfo(np.int32).min and np.max(outarr) <= np.iinfo(np.int32).max:

            outarr = outarr.astype(np.int32)

        else:
            raise TypeError('The interpolated data cube cannot be cast into a 16- or 32-bit integer array.')

        assert outarr.shape == tuple([*arrcube.shape[:2], len(target_CWLs)])

        return GeoArray(outarr)

    def predict_by_machine_learner(self, arrcube, method, src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor,
                                   tgt_LBA, n_clusters=50, classif_alg='MinDist', kNN_n_neighbors=10,
                                   global_clf_threshold=options['classifiers']['prediction']['global_clf_threshold'],
                                   src_nodataVal=None, out_nodataVal=None, compute_errors=False, bandwise_errors=True,
                                   fallback_argskwargs=None):
        # type: (Union[np.ndarray, GeoArray], str, str, str, list, str, str, list, int, str, int, Union[str, int, float], int, int, bool, bool, dict) -> tuple  # noqa
        """Predict spectral bands of target sensor by applying a machine learning approach.

        NOTE:   You may use the function spechomo.utils.list_available_transformations() to get a list of available
                transformations. You may also copy the input parameters for this method from the output there.

        :param arrcube:             input image array for target sensor spectral band prediction (rows x cols x bands)
        :param method:              machine learning approach to be used for spectral bands prediction
                                    'LR': Linear Regression
                                    'RR': Ridge Regression
                                    'QR': Quadratic Regression
                                    'RFR':  Random Forest Regression  (50 trees; does not allow spectral sub-clustering)
        :param src_satellite:       source satellite, e.g., 'Landsat-8'
        :param src_sensor:          source sensor, e.g., 'OLI_TIRS'
        :param src_LBA:             source LayerBandsAssignment  # TODO document this
        :param tgt_satellite:       target satellite, e.g., 'Landsat-8'
        :param tgt_sensor:          target sensor, e.g., 'OLI_TIRS'
        :param tgt_LBA:             target LayerBandsAssignment  # TODO document this
        :param n_clusters:          Number of spectral clusters to be used during LR/ RR/ QR homogenization.
                                    E.g., 50 means that the image to be converted to the spectral target sensor
                                    is clustered into 50 spectral clusters and one separate machine learner per
                                    cluster is applied to the input data to predict the homogenized image. If
                                    'spechomo_n_clusters' is set to 1, the source image is not clustered and
                                    only one machine learning classifier is used for prediction.
        :param classif_alg:         Multispectral classification algorithm to be used to determine the spectral cluster
                                    each pixel belongs to.
                                    'MinDist': Minimum Distance (Nearest Centroid)
                                    'kNN': k-nearest-neighbour
                                    'kNN_MinDist': k-nearest-neighbour Minimum Distance (Nearest Centroid)
                                    'SAM': spectral angle mapping
                                    'kNN_SAM': k-nearest-neighbour spectral angle mapping
                                    'SID': spectral information divergence
                                    'FEDSA': fused euclidian distance / spectral angle
                                    'kNN_FEDSA': k-nearest-neighbour fused euclidian distance / spectral angle
        :param kNN_n_neighbors:     The number of neighbors to be considered in case 'classif_alg' is set to 'kNN'.
                                    Otherwise, this parameter is ignored.
        :param global_clf_threshold:  If given, all pixels where the computed similarity metric (set by 'classif_alg')
                                      exceeds the given threshold are predicted using the global classifier (based on a
                                      single transformation per band).
                                      - only usable for 'MinDist', 'SAM' and 'SID'
                                      - may be given as float, integer or string to label a certain distance percentile
                                      - if given as string, it must match the format, e.g., '10%' for labelling the
                                      worst 10 % of the distances as unclassified
        :param src_nodataVal:       no data value of source image (arrcube)
                                    - if no nodata value is set, it is tried to be auto-computed from arrcube
        :param out_nodataVal:       no data value of predicted image
        :param compute_errors:      whether to compute pixel- / bandwise model errors for estimated pixel values
                                    (default: false)
        :param bandwise_errors      whether to compute error information for each band separately (True - default)
                                    or to average errors over bands using median (False) (ignored in case of fallback)
        :param fallback_argskwargs: arguments and keyword arguments to be passed to the fallback algorithm
                                    SpectralHomogenizer.interpolate_cube() in case harmonization fails
        :return:                    predicted array (rows x columns x bands)
        :rtype:                     Tuple[np.ndarray, Union[np.ndarray, None]]
        """
        # TODO: add LBA validation to .predict()
        kw = dict(method=method,
                  classifier_rootDir=self.classifier_rootDir,
                  n_clusters=n_clusters,
                  classif_alg=classif_alg,
                  CPUs=self.CPUs,
                  logger=self.logger)

        if classif_alg.startswith('kNN'):
            kw['n_neighbors'] = kNN_n_neighbors

        RSI_CP = RSImage_ClusterPredictor(**kw)

        ######################
        # get the classifier #
        ######################

        cls = None
        exc = Exception()
        try:
            cls = RSI_CP.get_classifier(src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA)

        except FileNotFoundError as e:
            self.logger.warning('No machine learning classifier available that fulfills the specifications of the '
                                'spectral reference sensor. Falling back to linear interpolation for performing '
                                'spectral homogenization.')
            exc = e

        except ClassifierNotAvailableError as e:
            self.logger.error('\nAn error occurred during spectral homogenization using the %s classifier. '
                              'Falling back to linear interpolation. Error message was: ' % method)
            self.logger.error(traceback.format_exc())
            exc = e

        ##################
        # run prediction #
        ##################

        errors = None
        if cls:
            self.logger.info('Performing spectral homogenization using %s. Target is %s %s %s.'
                             % (method, tgt_satellite, tgt_sensor, tgt_LBA))
            im_homo = RSI_CP.predict(arrcube,
                                     classifier=cls,
                                     in_nodataVal=src_nodataVal,
                                     cmap_nodataVal=src_nodataVal,
                                     out_nodataVal=out_nodataVal,
                                     global_clf_threshold=global_clf_threshold)  # type: GeoArray

            if compute_errors:
                errors = RSI_CP.compute_prediction_errors(im_homo, cls,
                                                          nodataVal=src_nodataVal,
                                                          cmap_nodataVal=src_nodataVal)

                if not bandwise_errors:
                    errors = np.median(errors, axis=2).astype(errors.dtype)

        elif fallback_argskwargs:
            # fallback: use linear interpolation and set errors to an array of zeros
            im_homo = self.interpolate_cube(**fallback_argskwargs)  # type: GeoArray

            if compute_errors:
                self.logger.warning("Spectral homogenization algorithm had to be performed by linear interpolation "
                                    "(fallback). Unable to compute any accuracy information from that.")
                if bandwise_errors:
                    errors = np.zeros_like(im_homo, dtype=np.int16)
                else:
                    errors = np.zeros(im_homo.shape[:2], dtype=np.int16)

        else:
            raise exc

        # add metadata
        im_homo.metadata.band_meta['wavelength'] = cls.tgt_wavelengths if cls else fallback_argskwargs['target_CWLs']
        im_homo.classif_map = RSI_CP.classif_map
        im_homo.distance_metrics = RSI_CP.distance_metrics

        # handle negative values in the predicted image => set these pixels to nodata
        # im_homo = set_negVals_to_nodata(im_homo, out_nodataVal)

        return im_homo, errors


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
                                    'kNN_MinDist': k-nearest-neighbour Minimum Distance (Nearest Centroid)
                                    'SAM': spectral angle mapping
                                    'kNN_SAM': k-nearest-neighbour spectral angle mapping
                                    'SID': spectral information divergence
                                    'FEDSA': fused euclidian distance / spectral angle
                                    'kNN_FEDSA': k-nearest-neighbour fused euclidian distance / spectral angle
        :param classifier_rootDir:  root directory where machine learning classifiers are stored.
        :param CPUs:                number of CPUs to use (default: 1)
        :param logger:              instance of logging.Logger()
        :param kw_clf_init          keyword arguments to be passed to classifier init functions if possible,
                                    e.g., 'n_neighbours' sets the number of neighbours to be considered in kNN
                                    classification algorithms (set by 'classif_alg')
        """
        self.method = method
        self.n_clusters = n_clusters
        self.classifier_rootDir = os.path.abspath(classifier_rootDir) if classifier_rootDir else _classifier_rootdir
        self.classif_map = None
        self.classif_map_fractions = None
        self.distance_metrics = None
        self.CPUs = CPUs or cpu_count()
        self.classif_alg = classif_alg
        self.logger = logger or SpecHomo_Logger(__name__)  # must be pickable
        self.kw_clf_init = kw_clf_init

        # validate
        if method == 'RFR' and n_clusters > 1:
            self.logger.warning("The spectral homogenization method 'Random Forest Regression' does not allow spectral "
                                "sub-clustering. Setting 'n_clusters' to 1.")
            self.n_clusters = 1

        if self.classif_alg.startswith('kNN') and \
           'n_neighbors' in kw_clf_init and \
           self.n_clusters < kw_clf_init['n_neighbors']:
            self.kw_clf_init['n_neighbors'] = self.n_clusters

    def get_classifier(self, src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA):
        # type: (str, str, list, str, str, list) -> Cluster_Learner
        """Select the correct machine learning classifier out of previously saved classifier collections.

        Describe the classifier specifications with the given arguments.
        :param src_satellite:   source satellite, e.g., 'Landsat-8'
        :param src_sensor:      source sensor, e.g., 'OLI_TIRS'
        :param src_LBA:         source LayerBandsAssignment
        :param tgt_satellite:   target satellite, e.g., 'Landsat-8'
        :param tgt_sensor:      target sensor, e.g., 'OLI_TIRS'
        :param tgt_LBA:         target LayerBandsAssignment
        :return:                classifier instance loaded from disk
        """
        args_fd = (self.classifier_rootDir, self.method, self.n_clusters,
                   src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA)

        try:
            CL = Cluster_Learner.from_disk(*args_fd)

        except FileNotFoundError:
            if self.classifier_rootDir == _classifier_rootdir:
                # the default root directory is used

                if not os.path.exists(os.path.join(_classifier_rootdir, '%s_classifiers.zip' % self.method)):
                    # download the classifiers
                    self.logger.info('The pre-trained classifiers have not been downloaded yet. Downloading...')

                    from .utils import download_pretrained_classifiers
                    download_pretrained_classifiers(method=self.method,
                                                    tgt_dir=self.classifier_rootDir)

                else:
                    self.logger.error('%s classifiers found at %s. However, they do not contain a suitable classifier '
                                      'for the current predition. If desired, delete the existing classifiers and try '
                                      'again. Pre-trained classifiers are then automatically downloaded.'
                                      % (self.method, self.classifier_rootDir))

                # try again
                CL = Cluster_Learner.from_disk(*args_fd)

            else:
                # classifier not found in the user provided root directory
                raise

        return CL

    def predict(self, image, classifier, in_nodataVal=None, out_nodataVal=None, cmap_nodataVal=None,
                global_clf_threshold=None, unclassified_pixVal=-1):
        # type: (Union[np.ndarray, GeoArray], Cluster_Learner, float, float, float, Union[str, int, float], int) -> GeoArray  # noqa
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
        :param global_clf_threshold:  If given, all pixels where the computed similarity metric (set by 'classif_alg')
                                      exceeds the given threshold are predicted using the global classifier (based on a
                                      single transformation per band).
                                      - not usable for 'kNN'
                                      - may be given as float, integer or string to label a certain distance percentile
                                      - if given as string, it must match the format, e.g., '10%' for labelling the
                                      worst 10 % of the distances as unclassified
        :param unclassified_pixVal:     pixel value to be used in the classification map for unclassified pixels
                                        (default: -1)
        :return:                3D array representing the predicted spectral image cube
        """
        image = image if isinstance(image, GeoArray) else GeoArray(image, nodata=in_nodataVal)

        # ensure image.nodata is present (important for classify_image() -> overwrites cmap at nodata positions)
        image.nodata = in_nodataVal if in_nodataVal is not None else image.nodata  # might be auto-computed here

        ##########################
        # get classification map #
        ##########################

        # assign each input pixel to a cluster (compute classification with cluster centers as endmembers)
        if self.classif_map is None:
            if self.n_clusters > 1:
                t0 = time.time()
                kw_clf = dict(classif_alg=self.classif_alg,
                              in_nodataVal=image.nodata,
                              cmap_nodataVal=cmap_nodataVal,  # written into classif_map at nodata
                              CPUs=self.CPUs,
                              return_distance=True,
                              **self.kw_clf_init)

                if self.classif_alg in ['MinDist', 'kNN_MinDist', 'SAM', 'kNN_SAM', 'SID', 'FEDSA', 'kNN_FEDSA']:
                    kw_clf.update(dict(unclassified_threshold=global_clf_threshold,
                                       unclassified_pixVal=unclassified_pixVal))

                if self.classif_alg == 'RF':
                    train_spectra = np.vstack([classifier.MLdict[clust].cluster_sample_spectra
                                               for clust in range(classifier.n_clusters)])
                    train_labels = list(np.hstack([[i] * 100
                                                   for i in range(classifier.n_clusters)]))
                else:
                    train_spectra = classifier.cluster_centers
                    train_labels = classifier.cluster_pixVals

                self.classif_map, self.distance_metrics = classify_image(image, train_spectra, train_labels, **kw_clf)

                # compute spectral distance
                # dist = kNN_MinimumDistance_Classifier.compute_euclidian_distance_3D(image, train_spectra)
                # idxs = self.classif_map.reshape(-1, self.classif_map.shape[2])
                # self.distance_metrics = \
                #     dist.reshape(-1, dist.shape[2])[np.arange(dist.shape[0] * dist.shape[1])[:, np.newaxis], idxs] \
                #         .reshape(self.classif_map.shape)
                # print('ED MAX MIN:', self.distance_metrics.max(), self.distance_metrics.min())

                self.logger.info('Total classification time: %s'
                                 % time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)))

            else:
                self.classif_map = GeoArray(np.full((image.rows,
                                                     image.cols),
                                                    classifier.cluster_pixVals[0],
                                                    np.int16),
                                            nodata=cmap_nodataVal)

                # overwrite all pixels where the input image contains nodata in ANY band
                # (would lead to faulty predictions due to multivariate prediction algorithms)
                if in_nodataVal is not None and cmap_nodataVal is not None:
                    self.classif_map[np.any(image[:] == image.nodata, axis=2)] = cmap_nodataVal

                self.distance_metrics = np.zeros_like(self.classif_map,
                                                      np.float32)

        ####################
        # apply prediction #
        ####################

        # adjust classifier
        if self.CPUs is None or self.CPUs > 1:
            # FIXME does not work -> parallelize with https://github.com/ajtulloch/sklearn-compiledtrees?
            classifier.n_jobs = cpu_count() if self.CPUs is None else self.CPUs

        # NOTE: prediction is applied in 1000 x 1000 tiles to save memory (because classifier.predict returns float32)
        t0 = time.time()
        out_nodataVal = out_nodataVal if out_nodataVal is not None else image.nodata
        image_predicted = GeoArray(np.empty((image.rows,
                                             image.cols,
                                             classifier.tgt_n_bands),
                                            dtype=image.dtype),
                                   geotransform=image.gt,
                                   projection=image.prj,
                                   nodata=out_nodataVal,
                                   bandnames=['B%s' % i
                                              if len(i) == 2
                                              else 'B0%s' % i
                                              for i in classifier.tgt_LBA])

        if classifier.n_clusters > 1 and\
           self.classif_map.ndim > 2:

            dist_min, dist_max = np.min(self.distance_metrics),\
                                 np.max(self.distance_metrics)
            dist_norm = (self.distance_metrics - dist_min) /\
                        (dist_max - dist_min)
            weights = 1 - dist_norm

        else:
            weights = None

        # weights = None if self.classif_map.ndim == 2 else \
        #     1 - (self.distance_metrics / np.sum(self.distance_metrics, axis=2, keepdims=True))

        # if self.classif_map.ndim > 2:
        #     print(self.distance_metrics[0, 0, :])
        #     print(weights[0, 0, :])

        n_saturated_px = 0
        for ((rS, rE), (cS, cE)), im_tile in image.tiles(tilesize=(1000, 1000)):
            self.logger.info('Predicting tile ((%s, %s), (%s, %s))...' % (rS, rE, cS, cE))

            classif_map_tile = self.classif_map[rS: rE + 1, cS: cE + 1]  # integer array

            # predict!
            if self.classif_map.ndim == 2:
                im_tile_pred = \
                    classifier.predict(im_tile, classif_map_tile,
                                       nodataVal=out_nodataVal,
                                       cmap_nodataVal=cmap_nodataVal,
                                       cmap_unclassifiedVal=unclassified_pixVal)

            else:
                weights_tile = weights[rS: rE + 1, cS: cE + 1]  # float array

                im_tile_pred = \
                    classifier.predict_weighted_averages(im_tile, classif_map_tile, weights_tile,
                                                         nodataVal=out_nodataVal,
                                                         cmap_nodataVal=cmap_nodataVal,
                                                         cmap_unclassifiedVal=unclassified_pixVal)

            # set saturated pixels (exceeding the output data range with respect to the data type) to no-data
            if isinstance(image_predicted.dtype, np.integer):
                out_dTMin, out_dTMax = np.iinfo(image_predicted.dtype).min,\
                                       np.iinfo(image_predicted.dtype).max

                if np.min(im_tile_pred) < out_dTMin or\
                   np.max(im_tile_pred) > out_dTMax:

                    mask_saturated = np.any(im_tile_pred > out_dTMax |
                                            im_tile_pred < out_dTMin,
                                            axis=2)
                    n_saturated_px += np.sum(mask_saturated)
                    im_tile_pred[mask_saturated] = out_nodataVal

            image_predicted[rS:rE + 1, cS:cE + 1] = im_tile_pred.astype(image_predicted.dtype)

        if n_saturated_px:
            self.logger.warning("%.2f %% of the predicted pixels are saturated and set to no-data."
                                % n_saturated_px / np.dot(*image_predicted.shape[:2]) * 100)

        # TODO add multiprocessing here? ML classifiers seem to use multiprocessing already
        # print(time.time() -t0)
        # t0 = time.time()
        # from multiprocessing import Pool
        # from geoarray.baseclasses import get_array_tilebounds
        # with Pool(self.CPUs, initializer=_mp_initializer, initargs=(image, self.classif_map, classifier)) as pool:
        #     tiles_pred = pool.starmap(_predict_tile_mp,
        #                               [(tilebounds, out_nodataVal, cmap_nodataVal)
        #                                for tilebounds in get_array_tilebounds(array_shape=image.shape,
        #                                                                       tile_shape=(1000, 1000))])
        #
        # for ((rS, rE), (cS, cE)), tile_pred in tiles_pred:
        #     image_predicted[rS: rE + 1, cS: cE + 1, :] = tile_pred
        #
        # print(time.time() - t0)

        self.logger.info('Total prediction time: %s' % time.strftime("%H:%M:%S", time.gmtime(time.time()-t0)))

        ###############################
        # complete prediction results #
        ###############################

        # re-apply nodata values to predicted result
        if image.nodata is not None:
            mask_nodata = image.calc_mask_nodata(overwrite=True, flag='any')
            image_predicted[~mask_nodata] = out_nodataVal

        # copy mask_nodata
        image_predicted.mask_nodata = image.mask_nodata

        # image_predicted.save(
        #     '/home/gfz-fe/scheffler/temp/SPECHOM_py/image_predicted_QRclust1_MinDist_noB9.bsq')
        # GeoArray(self.classif_map).save(
        #     '/home/gfz-fe/scheffler/temp/SPECHOM_py/classif_map_QRclust1_MinDist_noB9.bsq')

        # append some statistics regarding the homogenization
        cmap_vals, cmap_valcounts = np.unique(self.classif_map, return_counts=True)
        cmap_valfractions = cmap_valcounts / self.classif_map.size
        self.classif_map_fractions = dict(zip(list(cmap_vals), list(cmap_valfractions)))

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

        for clf in cluster_classifier:
            if not len(clf.rmse_per_band) == GeoArray(im_predicted).bands:
                raise ValueError('The given classifier contains error statistics incompatible to the shape of the '
                                 'image.')
        if self.classif_map is None:
            raise RuntimeError('self.classif_map must be generated by running self.predict() beforehand.')

        if self.classif_map.ndim == 3:
            # FIXME: error computation does not work for kNN algorithms so far (self.classif_map is 3D instead of 2D)
            raise NotImplementedError('Error computation for 3-dimensional classification maps (e.g., due to kNN '
                                      'classification algorithms) is not yet implemented.')

        errors = np.empty_like(im_predicted)

        # iterate over all cluster labels and copy rmse values
        for pixVal in sorted(list(np.unique(self.classif_map))):
            if pixVal == cmap_nodataVal:
                continue

            self.logger.info('Inpainting error values for cluster #%s...' % pixVal)

            clf2use = cluster_classifier.MLdict[pixVal] if pixVal != -1 else cluster_classifier.global_clf
            rmse_per_band_int = np.round(clf2use.rmse_per_band, 0).astype(np.int16)
            errors[self.classif_map[:] == pixVal] = rmse_per_band_int

        # TODO validate this equation
        # errors = (errors * im_predicted[:] / 10000).astype(errors.dtype)

        # re-apply nodata values to predicted result
        if im_predicted.nodata is not None:
            # errors[im_predicted == im_predicted.nodata] = im_predicted.nodata
            errors[im_predicted.mask_nodata.astype(np.int8) == 0] = im_predicted.nodata

        # GeoArray(errors).save('/home/gfz-fe/scheffler/temp/SPECHOM_py/errors_LRclust1_MinDist_noB9_clusterpred.bsq')

        return errors

#
# _global_image, _global_classif_map, _global_classifier = None, None, None
#
#
# def _mp_initializer(image, classif_map, classifier):
#     global _global_image, _global_classif_map, _global_classifier
#     _global_image, _global_classif_map, _global_classifier = image, classif_map, classifier
#
#
# def _predict_tile_mp(tilebounds, out_nodataVal, cmap_nodataVal):
#     (rS, rE), (cS, cE) = tilebounds
#     im_tile = _global_image[rS: rE + 1, cS: cE + 1, :]
#     classif_map_tile = _global_classif_map[rS: rE + 1, cS: cE + 1]  # integer array
#     classifier = _global_classifier
#
#     # predict!
#     im_tile_pred = \
#         classifier.predict(im_tile, classif_map_tile,
#                            nodataVal=out_nodataVal, cmap_nodataVal=cmap_nodataVal).astype(_global_image.dtype)
#
#     return tilebounds, im_tile_pred
