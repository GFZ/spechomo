# -*- coding: utf-8 -*-

import os
import re
from glob import glob
from multiprocessing import cpu_count
from typing import List, Tuple, Union, Dict  # noqa F401  # flake8 issue
import logging  # noqa F401  # flake8 issue

import dill
import numpy as np
from nested_dict import nested_dict
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from geoarray import GeoArray

from .clustering import KMeansRSImage
from .prediction import im2spectra
from spechomo.resampling import SpectralResampler
from .training_data import RefCube


class ReferenceCube_Generator(object):
    """Class for creating reference cube that are later used as training data for SpecHomo_Classifier."""

    def __init__(self, filelist_refs, tgt_sat_sen_list=None, dir_refcubes='', n_clusters=10, tgt_n_samples=1000,
                 v=False, logger=None, CPUs=None):
        # type: (List[str], List[Tuple[str, str]], str, int, int, bool, logging.Logger, Union[None, int]) -> None
        """Initialize ReferenceCube_Generator.

        :param filelist_refs:   list of (hyperspectral) reference images,
                                representing BOA reflectance, scaled between 0 and 10000
        :param tgt_sat_sen_list:    list satellite/sensor tuples containing those sensors for which the reference cube
                                    is to be computed, e.g. [('Landsat-8', 'OLI_TIRS',), ('Landsat-5', 'TM')]
        :param dir_refcubes:    output directory for the generated reference cube
        :param n_clusters:      number of clusters to be used for clustering the input images (KMeans)
        :param tgt_n_samples:   number o spectra to be collected from each input image
        :param v:               verbose mode
        :param logger:          instance of logging.Logger()
        :param CPUs:            number CPUs to use for computation
        """
        # args + kwargs
        self.ims_ref = [filelist_refs, ] if isinstance(filelist_refs, str) else filelist_refs
        self.tgt_sat_sen_list = tgt_sat_sen_list or [
            ('Landsat-8', 'OLI_TIRS'),
            ('Landsat-7', 'ETM+'),
            ('Landsat-5', 'TM'),
            ('Sentinel-2A', 'MSI'),
            # ('Terra', 'ASTER'),  # currently does not work
            ('SPOT-4', 'HRVIR1'),
            ('SPOT-4', 'HRVIR2'),
            ('SPOT-5', 'HRG1'),
            ('SPOT-5', 'HRG2'),
            ('RapidEye-5', 'MSI')
            ]
        self.dir_refcubes = os.path.abspath(dir_refcubes) if dir_refcubes else ''
        self.n_clusters = n_clusters
        self.tgt_n_samples = tgt_n_samples
        self.v = v
        self.logger = logger or GMS_logger(__name__)  # must be pickable
        self.CPUs = CPUs or cpu_count()

        # privates
        self._refcubes = \
            {(sat, sen): RefCube(satellite=sat, sensor=sen,
                                 LayerBandsAssignment=self._get_tgt_LayerBandsAssignment(sat, sen))
             for sat, sen in self.tgt_sat_sen_list}

        # validation
        if dir_refcubes and not os.path.isdir(self.dir_refcubes):
            raise ValueError("%s is not a directory." % self.dir_refcubes)

    def __getstate__(self):
        """Defines how the attributes of ReferenceCube_Generator instances are pickled."""
        close_logger(self.logger)
        self.logger = None

        return self.__dict__

    def __del__(self):
        close_logger(self.logger)
        self.logger = None

    @property
    def refcubes(self):
        # type: () -> Dict[Tuple[str, str]: RefCube]
        """Return a dictionary holding instances of RefCube for each target satellite / sensor of self.tgt_sat_sen_list.
        """
        if not self._refcubes:

            # fill self._ref_cubes with GeoArray instances of already existing reference cubes read from disk
            if self.dir_refcubes:
                for path_refcube in glob(os.path.join(self.dir_refcubes, 'refcube__*.bsq')):
                    # TODO check if that really works
                    # check if current refcube path matches the target refcube specifications
                    identifier = re.search('refcube__(.*).bsq', os.path.basename(path_refcube)).group(1)
                    sat, sen, nclust_str, nsamp_str = identifier.split('__')  # type: str
                    nclust, nsamp = int(nclust_str.split('nclust')[1]), int(nsamp_str.split('nclust')[1])
                    correct_specs = all([(sat, sen) in self.tgt_sat_sen_list,
                                         nclust == self.n_clusters,
                                         nsamp == self.tgt_n_samples])

                    # import the existing ref cube if it matches the target refcube specs
                    if correct_specs:
                        self._refcubes[(sat, sen)] = \
                            RefCube(satellite=sat, sensor=sen, filepath=path_refcube,
                                    LayerBandsAssignment=self._get_tgt_LayerBandsAssignment(sat, sen))

        return self._refcubes

    def _get_tgt_GMS_identifier(self, tgt_sat, tgt_sen):
        # type: (str, str) -> GMS_identifier
        """Get a GMS identifier for the specified target sensor such that all possible bands are included (L1A)

        :param tgt_sat:     target satellite
        :param tgt_sen:     target sensor
        :return:
        """
        return GMS_identifier(satellite=tgt_sat, sensor=tgt_sen, subsystem=None, image_type='RSD', dataset_ID=-9999,
                              proc_level='L1A', logger=self.logger)  # use L1A to have all bands available

    def _get_tgt_LayerBandsAssignment(self, tgt_sat, tgt_sen):
        # type: (str, str) -> list
        """Get the LayerBandsAssignment for the specified target sensor.

        NOTE:   The returned bands list always contains all possible bands. Specific band seletions are later done
                using np.take().

        :param tgt_sat:     target satellite
        :param tgt_sen:     target sensor
        :return:
        """
        return get_LayerBandsAssignment(self._get_tgt_GMS_identifier(tgt_sat, tgt_sen), no_pan=False)

    def _get_tgt_SRF_object(self, tgt_sat, tgt_sen):
        # type: (str, str) -> SRF
        """Get an SRF instance containing the spectral response functions for for the specified target sensor.

        :param tgt_sat:     target satellite
        :param tgt_sen:     target sensor
        :return:
        """
        return SRF(self._get_tgt_GMS_identifier(tgt_sat, tgt_sen), no_pan=False)

    def generate_reference_cubes(self, fmt_out='ENVI', progress=True):
        # type: (str, bool) -> ReferenceCube_Generator.refcubes
        """Generate reference spectra from all hyperspectral input images.

        Workflow:
        1. Clustering/classification of hyperspectral images and selection of a given number of random signatures
            (a. Spectral downsamling to lower spectral resolution (speedup))
            b. KMeans clustering
            c. Selection of the same number of signatures from each cluster to avoid unequal amount of training data.
        2. Spectral resampling of the selected hyperspectral signatures (for each input image)
        3. Add resampled spectra to reference cubes for each target sensor and write cubes to disk

        :param fmt_out:         output format (GDAL driver code)
        :param progress:        show progress bar (default: True)
        :return:                np.array: [tgt_n_samples x images x spectral bands of the target sensor]
        """
        for im in self.ims_ref:
            # TODO implement check if current image is already included in the refcube -> skip in that case
            src_im = GeoArray(im)

            # get random spectra of the original (hyperspectral) image, equally distributed over all computed clusters
            unif_random_spectra = self.cluster_image_and_get_uniform_spectra(src_im, progress=progress).astype(np.int16)

            # resample the set of random spectra to match the spectral characteristics of all target sensors
            for tgt_sat, tgt_sen in self.tgt_sat_sen_list:
                # perform spectral resampling
                self.logger.info('Performing spectral resampling to match %s %s specifications...' % (tgt_sat, tgt_sen))
                unif_random_spectra_rsp = self.resample_spectra(
                    unif_random_spectra,
                    src_cwl=np.array(src_im.meta.band_meta['wavelength'], dtype=np.float).flatten(),
                    tgt_srf=self._get_tgt_SRF_object(tgt_sat, tgt_sen))

                # add the spectra as GeoArray instance to the in-mem ref cubes
                refcube = self.refcubes[(tgt_sat, tgt_sen)]  # type: RefCube
                refcube.add_spectra(unif_random_spectra_rsp, src_imname=src_im.basename,
                                    LayerBandsAssignment=self._get_tgt_LayerBandsAssignment(tgt_sat, tgt_sen))

                # update the reference cubes on disk
                if self.dir_refcubes:
                    refcube.save(path_out=os.path.join(self.dir_refcubes, 'refcube__%s__%s__nclust%s__nsamp%s.bsq'
                                                       % (tgt_sat, tgt_sen, self.n_clusters, self.tgt_n_samples)),
                                 fmt=fmt_out)

        return self.refcubes

    def cluster_image_and_get_uniform_spectra(self, im, downsamp_sat='Sentinel-2A', downsamp_sen='MSI', progress=False):
        # type: (Union[str, GeoArray, np.ndarray], str, str, bool) -> np.ndarray
        """Compute KMeans clusters for the given image and return the an array of uniform random samples.

        :param im:              image to be clustered
        :param downsamp_sat:    satellite code used for intermediate image dimensionality reduction (input image is
                                spectrally resampled to this satellite before it is clustered). requires downsamp_sen.
                                If it is None, no intermediate downsampling is performed.
        :param downsamp_sen:    sensor code used for intermediate image dimensionality reduction (required downsamp_sat)
        :param progress:        whether to show progress bars or not
        :return:    2D array (rows: tgt_n_samples, columns: spectral information / bands
        """
        # input validation
        if downsamp_sat and not downsamp_sen or downsamp_sen and not downsamp_sat:
            raise ValueError("The parameters 'spec_rsp_sat' and 'spec_rsp_sen' must both be provided or completely "
                             "omitted.")

        im2clust = GeoArray(im)

        # first, perform spectral resampling to Sentinel-2 to reduce dimensionality (speedup)
        if downsamp_sat and downsamp_sen:
            tgt_srf = SRF(GMS_identifier(satellite=downsamp_sat, sensor=downsamp_sen, subsystem=None, image_type='RSD',
                                         dataset_ID=-9999, proc_level='L1A', logger=self.logger))
            im2clust = self.resample_image_spectrally(im2clust, tgt_srf, progress=progress)  # output = int16

        # compute KMeans clusters for the spectrally resampled image
        self.logger.info('Computing %s KMeans clusters from the input image %s...'
                         % (self.n_clusters, im2clust.basename))
        kmeans = KMeansRSImage(im2clust, n_clusters=self.n_clusters, CPUs=self.CPUs, v=self.v)
        kmeans.compute_clusters()

        if self.v:
            kmeans.plot_cluster_centers()
            kmeans.plot_cluster_histogram()

        # randomly grab the given number of spectra from each cluster
        self.logger.info('Getting %s random spectra from each cluster...' % (self.tgt_n_samples // self.n_clusters))
        random_samples = kmeans.get_random_spectra_from_each_cluster(src_im=GeoArray(im),
                                                                     samplesize=self.tgt_n_samples // self.n_clusters)

        # combine the spectra (2D arrays) of all clusters to a single 2D array
        self.logger.info('Combining random samples from all clusters.')
        random_samples = np.vstack([random_samples[clusterlabel] for clusterlabel in random_samples])

        return random_samples

    def resample_spectra(self, spectra, src_cwl, tgt_srf):
        # type: (Union[GeoArray, np.ndarray], Union[list, np.array], SRF) -> np.ndarray
        """Perform spectral resampling of the given image to match the given spectral response functions.

        :param spectra:     2D array (rows: spectral samples;  columns: spectral information / bands
        :param src_cwl:     central wavelength positions of input spectra
        :param tgt_srf:     target spectral response functions to be used for spectral resampling
        :return:
        """
        spectra = GeoArray(spectra)

        # perform spectral resampling of input image to match spectral properties of target sensor
        self.logger.info('Performing spectral resampling to match spectral properties of %s %s...'
                         % (tgt_srf.satellite, tgt_srf.sensor))

        SR = SpectralResampler(src_cwl, tgt_srf)
        spectra_rsp = SR.resample_spectra(spectra, chunksize=200, CPUs=self.CPUs)

        return spectra_rsp

    def resample_image_spectrally(self, src_im, tgt_srf, progress=False):
        # type: (Union[str, GeoArray], SRF, bool) -> Union[GeoArray, None]
        """Perform spectral resampling of the given image to match the given spectral response functions.

        :param src_im:      source image to be resampled
        :param tgt_srf:     target spectral response functions to be used for spectral resampling
        :param progress:    show progress bar (default: false)
        :return:
        """
        # handle src_im provided as file path or GeoArray instance
        if isinstance(src_im, str):
            im_name = os.path.basename(src_im)
            im_gA = GeoArray(src_im)
        else:
            im_name = src_im.basename
            im_gA = src_im

        # read input image
        self.logger.info('Reading the input image %s...' % im_name)
        im_gA.cwl = np.array(im_gA.meta.band_meta['wavelength'], dtype=np.float).flatten()

        # perform spectral resampling of input image to match spectral properties of target sensor
        self.logger.info('Performing spectral resampling to match spectral properties of %s %s...'
                         % (tgt_srf.satellite, tgt_srf.sensor))
        SR = SpectralResampler(im_gA.cwl, tgt_srf)

        tgt_im = GeoArray(np.zeros((*im_gA.shape[:2], len(tgt_srf.bands)), dtype=np.int16), im_gA.gt, im_gA.prj)
        tgt_im.meta.band_meta['wavelength'] = list(tgt_srf.wvl)
        tgt_im.bandnames = ['B%s' % i if len(i) == 2 else 'B0%s' % i for i in tgt_srf.LayerBandsAssignment]

        tiles = im_gA.tiles((1000, 1000))  # use tiles to save memory
        for ((rS, rE), (cS, cE)), tiledata in (tqdm(tiles) if progress else tiles):
            tgt_im[rS: rE + 1, cS: cE + 1, :] = SR.resample_image(tiledata.astype(np.int16), CPUs=self.CPUs)

        return tgt_im


class ClusterClassifier_Generator(object):
    """Class for creating collections of machine learning classifiers that can be used for spectral homogenization."""
    def __init__(self, list_refcubes, logger=None):
        # type: (List[Union[str, RefCube]], logging.Logger) -> None
        """Get an instance of Classifier_Generator.

        :param list_refcubes:   list of RefCube instances for which the classifiers are to be created.
        :param logger:          instance of logging.Logger()
        """
        self.refcubes = [RefCube(inRC) if isinstance(inRC, str) else inRC for inRC in list_refcubes]
        self.logger = logger or GMS_logger(__name__)  # must be pickable

    def __getstate__(self):
        """Defines how the attributes of ReferenceCube_Generator instances are pickled."""
        close_logger(self.logger)
        self.logger = None

        return self.__dict__

    def __del__(self):
        close_logger(self.logger)
        self.logger = None

    @staticmethod
    def _get_derived_LayerBandsAssignments(satellite, sensor):
        """Get a list of possible LayerBandsAssignments in which the spectral training data may be arranged.

        :param satellite:   satellite to return LayerBandsAssignments for
        :param sensor:      sensor to return LayerBandsAssignments for
        :return:            e.g. for Landsat-8 OLI_TIRS:
                            [['1', '2', '3', '4', '5', '9', '6', '7'],
                             ['1', '2', '3', '4', '5', '9', '6', '7', '8'],
                             ['1', '2', '3', '4', '5', '6', '7'],
                             ['1', '2', '3', '4', '5', '6', '7'], ...]
        """
        L1A_GMSid = GMS_identifier(satellite=satellite, sensor=sensor, subsystem=None, image_type='RSD',
                                   dataset_ID=-9999, proc_level='L1A', logger=None)
        # different numbers of bands after AC
        L1C_GMSid = GMS_identifier(satellite=satellite, sensor=sensor, subsystem=None, image_type='RSD',
                                   dataset_ID=-9999, proc_level='L1C', logger=None)

        LBAs = []
        for lba in [get_LayerBandsAssignment(L1A_GMSid, no_pan=False, sort_by_cwl=True),  # L1A_withPan_cwlSorted
                    get_LayerBandsAssignment(L1C_GMSid, no_pan=False, sort_by_cwl=True),  # L1C_withPan_cwlSorted
                    get_LayerBandsAssignment(L1A_GMSid, no_pan=True, sort_by_cwl=True),  # L1A_noPan_cwlSorted
                    get_LayerBandsAssignment(L1C_GMSid, no_pan=True, sort_by_cwl=True),  # L1A_noPan_cwlSorted
                    get_LayerBandsAssignment(L1A_GMSid, no_pan=False, sort_by_cwl=False),  # L1A_withPan_alphabetical
                    get_LayerBandsAssignment(L1C_GMSid, no_pan=False, sort_by_cwl=False),  # L1C_withPan_alphabetical
                    get_LayerBandsAssignment(L1A_GMSid, no_pan=True, sort_by_cwl=False),  # L1A_noPan_alphabetical
                    get_LayerBandsAssignment(L1C_GMSid, no_pan=True, sort_by_cwl=False),  # L1C_noPan_alphabetical
                    ]:
            if lba not in LBAs:
                LBAs.append(lba)

        return LBAs

    @staticmethod
    def train_machine_learner(train_X, train_Y, test_X, test_Y, method, **kwargs):
        # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict) -> Union[LinearRegression, Ridge, Pipeline, RandomForestRegressor]  # noqa E501 (line too long)
        """Use the given train and test data to train a machine learner and append some accuracy statistics.

        :param train_X:     reference training data
        :param train_Y:     target training data
        :param test_X:      reference test data
        :param test_Y:      target test data
        :param method:      type of machine learning classifiers to be included in classifier collections
                            'LR':   Linear Regression
                            'RR':   Ridge Regression
                            'QR':   Quadratic Regression
                            'RFR':  Random Forest Regression (50 trees)
        :param kwargs:      keyword arguments to be passed to the __init__() function of machine learners
        """
        ###################
        # train the model #
        ###################

        ML = get_machine_learner(method, **kwargs)
        ML.fit(train_X, train_Y)

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)

            # avoid division by 0
            if 0 in y_true:
                y_true = y_true.astype(np.float)
                y_true[y_true == 0] = np.nan

            return np.nanmean(np.abs((y_true - y_pred) / y_true), axis=0) * 100

        ###########################
        # compute RMSE, MAE, MAPE #
        ###########################

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        predicted = ML.predict(test_X)  # returns 2D array (spectral samples x bands), e.g. 640 x 6
        # NOTE: 'raw_values': RMSE is column-wise computed
        #       => yields the same result as one would compute the RMSE band by band
        rmse = np.sqrt(mean_squared_error(test_Y, predicted, multioutput='raw_values')).astype(np.float32)
        mae = mean_absolute_error(test_Y, predicted, multioutput='raw_values').astype(np.float32)
        mape = mean_absolute_percentage_error(test_Y, predicted).astype(np.float32)

        # predicted_train = ML.predict(train_X)
        # rmse_train = np.sqrt(mean_squared_error(train_Y, predicted_train, multioutput='raw_values'))
        # # v2
        # test_Y_im = spectra2im(test_Y, tgt_rows=int(tgt_data.rows * 0.4), tgt_cols=tgt_data.cols)
        # pred_im = spectra2im(predicted, tgt_rows=int(tgt_data.rows * 0.4), tgt_cols=tgt_data.cols)
        # rmse_v2 = np.sqrt(mean_squared_error(test_Y_im[:,:,1], pred_im[:,:,1]))

        # append some metadata
        ML.scores = dict(train=ML.score(train_X, train_Y),
                         test=ML.score(test_X, test_Y))  # r2 scores
        ML.rmse_per_band = list(rmse)
        ML.mae_per_band = list(mae)
        ML.mape_per_band = list(mape)

        # convert float64 attributes to float32 to save memory (affects <0,05% of homogenized pixels by 1 DN)
        for attr in ['coef_', 'intercept_', 'singular_', '_residues']:
            if isinstance(ML, Pipeline):
                setattr(ML._final_estimator, attr, getattr(ML._final_estimator, attr).astype(np.float32))
            else:
                try:
                    setattr(ML, attr, getattr(ML, attr).astype(np.float32))
                except AttributeError:
                    pass

        return ML

    @staticmethod
    def cluster_refcube_spectra(cube2cluster, n_clusters=50, CPUs=24):
        # type: (Union[RefCube, str], int, int) -> np.ndarray
        """Generate cluster labels for the given reference cube.

        :param cube2cluster:    reference cube to be clustered
        :param n_clusters:      number of clusters to be used for KMeans clustering
        :param CPUs:            number of CPUs to be used for KMeans clustering
        """
        cube2cluster = RefCube(cube2cluster) if isinstance(cube2cluster, str) else cube2cluster

        # use this refcube for KMeans clustering
        km = KMeansRSImage(cube2cluster.data, n_clusters=n_clusters, CPUs=CPUs)
        km.compute_clusters()
        labels1D = km.clusters.labels_

        return labels1D

    def create_classifiers(self, outDir, method='LR', n_clusters=50, CPUs=24, **kwargs):
        # type: (str, str, int, int, dict) -> None
        """Create cluster classifiers for all combinations of the reference cubes given in __init__().

        :param outDir:      output directory for the created cluster classifier collections
        :param method:      type of machine learning classifiers to be included in classifier collections
                            'LR':   Linear Regression
                            'RR':   Ridge Regression
                            'QR':   Quadratic Regression
                            'RFR':  Random Forest Regression (50 trees)
        :param n_clusters:  number of clusters to be used for KMeans clustering
        :param CPUs:        number of CPUs to be used for KMeans clustering
        :param kwargs:      keyword arguments to be passed to machine learner
        """
        # validate and set defaults
        if method == 'RFR':
            if n_clusters > 1:
                self.logger.warning("The spectral homogenization method 'Random Forest Regression' does not allow "
                                    "spectral sub-clustering. Setting 'n_clusters' to 1.")
                n_clusters = 1

            if 'n_jobs' not in kwargs:
                kwargs.update(dict(n_jobs=CPUs))
            if 'n_estimators' not in kwargs:
                kwargs.update(dict(n_estimators=CFG.spechomo_rfr_n_trees))  # we statically use 50 trees for RFR
            if 'max_depth' not in kwargs:
                kwargs.update(dict(max_depth=CFG.spechomo_rfr_tree_depth))  # we statically use a maximum depth of 10

        # build the classifier collections with separate classifiers for each cluster
        for src_cube in self.refcubes:  # type: RefCube
            cls_collection = nested_dict()
            fName_cls = get_filename_classifier_collection(method, src_cube.satellite, src_cube.sensor,
                                                           n_clusters=n_clusters)

            # get cluster labels for each source cube separately
            self.logger.info('Clustering %s %s reference cube (%s clusters)...'
                             % (src_cube.satellite, src_cube.sensor, n_clusters))
            labels1D = self.cluster_refcube_spectra(src_cube, n_clusters=n_clusters, CPUs=CPUs)

            for tgt_cube in self.refcubes:
                if (src_cube.satellite, src_cube.sensor) == (tgt_cube.satellite, tgt_cube.sensor):
                    continue
                clf_str = 'classifier' if n_clusters == 1 else 'cluster classifier'
                self.logger.info("Creating %s %s to predict %s %s from %s %s..."
                                 % (method, clf_str, tgt_cube.satellite, tgt_cube.sensor,
                                    src_cube.satellite, src_cube.sensor))

                src_derived_LBAs = self._get_derived_LayerBandsAssignments(src_cube.satellite, src_cube.sensor)
                tgt_derived_LBAs = self._get_derived_LayerBandsAssignments(tgt_cube.satellite, tgt_cube.sensor)
                # possib_LBA_names = src_derived_LBAs.keys()  # would be the same for tgt_derived_LBA

                for src_LBA in src_derived_LBAs:
                    for tgt_LBA in tgt_derived_LBAs:
                        self.logger.debug('Creating %s cluster classifier for LBA %s => %s...'
                                          % (method, '_'.join(src_LBA), '_'.join(tgt_LBA)))

                        # Get center wavelength positions
                        # NOTE: they cannot be taken from RefCube instances because they always represent L1A LBAs
                        src_wavelengths = get_center_wavelengths_by_LBA(src_cube.satellite, src_cube.sensor, src_LBA)
                        tgt_wavelengths = get_center_wavelengths_by_LBA(tgt_cube.satellite, tgt_cube.sensor, tgt_LBA)

                        # Get training data for source and target image according to the given LayerBandsAssignments
                        # e.g., source: Landsat 7 image in LBA 1__2__3__4__5__7 and target L8 in 1__2__3__4__5__6__7
                        src_data = src_cube.get_band_combination(src_LBA)
                        tgt_data = tgt_cube.get_band_combination(tgt_LBA)

                        src_spectra = im2spectra(src_data)
                        tgt_spectra = im2spectra(tgt_data)

                        # assign clusterlabel to each spectrum
                        src_df = DataFrame(src_spectra, columns=['B%s' % band for band in src_LBA])
                        src_df.insert(0, 'cluster_label', labels1D)
                        tgt_df = DataFrame(tgt_spectra, columns=['B%s' % band for band in tgt_LBA])
                        tgt_df.insert(0, 'cluster_label', labels1D)

                        for clusterlabel in range(n_clusters):
                            self.logger.debug('Creating %s classifier for cluster %s...' % (method, clusterlabel))

                            # Set train and test variables
                            # NOTE: If random_state is set to an Integer,
                            #       train_test_split will always select the same 'pseudo-random' set of the input data.
                            src_spectra_label = np.array(src_df[src_df.cluster_label == clusterlabel])[:, 1:]
                            tgt_spectra_label = np.array(tgt_df[tgt_df.cluster_label == clusterlabel])[:, 1:]

                            train_X, test_X, train_Y, test_Y = \
                                train_test_split(src_spectra_label, tgt_spectra_label,
                                                 test_size=0.4, shuffle=True, random_state=0)

                            # compute cluster centers for source spectra (only on those spectra used for model training)
                            cluster_center = np.mean(train_X, axis=0).astype(src_data.dtype)

                            # train the learner
                            ML = self.train_machine_learner(train_X, train_Y, test_X, test_Y, method, **kwargs)

                            # add some metadata
                            ML.src_satellite = src_cube.satellite
                            ML.tgt_satellite = tgt_cube.satellite
                            ML.src_sensor = src_cube.sensor
                            ML.tgt_sensor = tgt_cube.sensor
                            ML.src_LBA = src_LBA
                            ML.tgt_LBA = tgt_LBA
                            ML.src_n_bands = len(ML.src_LBA)
                            ML.tgt_n_bands = len(ML.tgt_LBA)
                            ML.src_wavelengths = list(np.array(src_wavelengths).astype(np.float32))
                            ML.tgt_wavelengths = list(np.array(tgt_wavelengths).astype(np.float32))
                            ML.n_clusters = n_clusters
                            ML.clusterlabel = clusterlabel
                            ML.cluster_center = cluster_center

                            assert len(ML.src_LBA) == len(ML.src_wavelengths)
                            assert len(ML.tgt_LBA) == len(ML.tgt_wavelengths)

                            # append to classifier collection
                            cls_collection[
                                '__'.join(src_LBA)][tgt_cube.satellite, tgt_cube.sensor][
                                '__'.join(tgt_LBA)][clusterlabel] = ML

            # dump to disk
            with open(os.path.join(outDir, fName_cls), 'wb') as outF:
                dill.dump(cls_collection.to_dict(), outF)


def get_machine_learner(method='LR', **init_params):
    # type: (str, dict) -> Union[LinearRegression, Ridge, Pipeline]
    """Get an instance of a machine learner.

    :param method:          'LR':   Linear Regression
                            'RR':   Ridge Regression
                            'QR':   Quadratic Regression
                            'RFR':  Random Forest Regression (50 trees)
    :param init_params:     parameters to be passed to __init__() function of the returned machine learner model.
    """
    if method == 'LR':
        return LinearRegression(**init_params)
    elif method == 'RR':
        return Ridge(**init_params)
    elif method == 'QR':
        return make_pipeline(PolynomialFeatures(degree=2), LinearRegression(**init_params))
    elif method == 'RFR':
        return RandomForestRegressor(**init_params)
    else:
        raise ValueError("Unknown machine learner method code '%s'." % method)


def get_filename_classifier_collection(method, src_satellite, src_sensor, n_clusters=1):
    if method == 'RR':
        method += '_alpha1.0'  # TODO add to config
    elif method == 'RFR':
        method += '_trees%s' % CFG.spechomo_rfr_n_trees  # we statically use 50 trees for Random Forest Regression

    return '__'.join(['%s_clust%s' % (method, n_clusters), src_satellite, src_sensor]) + '.dill'


# def get_classifier_filename(method, src_satellite, src_sensor, src_LBA_name, tgt_satellite, tgt_sensor, tgt_LBA_name):
#     return '__'.join([method, src_satellite, src_sensor, src_LBA_name]) + \
#                     '__to__' + '__'.join([tgt_satellite, tgt_sensor, tgt_LBA_name]) + '.dill'
