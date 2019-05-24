# -*- coding: utf-8 -*-

import os
import re
from glob import glob
from multiprocessing import cpu_count
from typing import List, Tuple, Union, Dict, TYPE_CHECKING  # noqa F401  # flake8 issue
import logging  # noqa F401  # flake8 issue
from itertools import product

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
from .resampling import SpectralResampler
from .training_data import RefCube
from .logging import SpecHomo_Logger
from .options import options

# TODO dependencies to get rid of
if TYPE_CHECKING:
    from gms_preprocessing.model.gms_object import GMS_identifier
    # from gms_preprocessing.model.metadata import get_LayerBandsAssignment, get_center_wavelengths_by_LBA
    from gms_preprocessing.io.input_reader import SRF


class ReferenceCube_Generator(object):
    """Class for creating reference cube that are later used as training data for SpecHomo_Classifier."""

    def __init__(self, filelist_refs, tgt_sat_sen_list=None, dir_refcubes='', n_clusters=10, tgt_n_samples=1000,
                 v=False, logger=None, CPUs=None, dir_clf_dump=''):
        # type: (List[str], List[Tuple[str, str]], str, int, int, bool, logging.Logger, Union[None, int], str) -> None
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
        :param dir_clf_dump:    directory where to store the serialized KMeans classifier
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
        self.logger = logger or SpecHomo_Logger(__name__)  # must be pickable
        self.CPUs = CPUs or cpu_count()
        self.dir_clf_dump = dir_clf_dump

        # privates
        self._refcubes = \
            {(sat, sen): RefCube(satellite=sat, sensor=sen,
                                 LayerBandsAssignment=self._get_tgt_LayerBandsAssignment(sat, sen))
             for sat, sen in self.tgt_sat_sen_list}

        if dir_refcubes and not os.path.isdir(self.dir_refcubes):
            os.makedirs(self.dir_refcubes)

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
        """Get a GMS identifier for the specified target sensor such that all possible bands are included (L1C)

        NOTE: We don't use L1A here because the signal of the additional bands at L1A is not predictable by spectral
              harmonization as these bands are not driven by surface albedo but by atmospheric conditions (945, 1373nm).

        :param tgt_sat:     target satellite
        :param tgt_sen:     target sensor
        :return:
        """
        from gms_preprocessing.model.gms_object import GMS_identifier
        return GMS_identifier(satellite=tgt_sat, sensor=tgt_sen, subsystem=None, image_type='RSD', dataset_ID=-9999,
                              proc_level='L1C', logger=self.logger)  # use L1A to have all bands available

    def _get_tgt_LayerBandsAssignment(self, tgt_sat, tgt_sen):
        # type: (str, str) -> list
        """Get the LayerBandsAssignment for the specified target sensor.

        NOTE:   The returned bands list always contains all possible bands. Specific band seletions are later done
                using np.take().

        :param tgt_sat:     target satellite
        :param tgt_sen:     target sensor
        :return:
        """
        from gms_preprocessing.model.metadata import get_LayerBandsAssignment
        return get_LayerBandsAssignment(self._get_tgt_GMS_identifier(tgt_sat, tgt_sen), no_pan=False, sort_by_cwl=True)

    def _get_tgt_SRF_object(self, tgt_sat, tgt_sen):
        # type: (str, str) -> SRF
        """Get an SRF instance containing the spectral response functions for for the specified target sensor.

        :param tgt_sat:     target satellite
        :param tgt_sen:     target sensor
        :return:
        """
        from gms_preprocessing.io.input_reader import SRF
        return SRF(self._get_tgt_GMS_identifier(tgt_sat, tgt_sen), no_pan=False)

    def generate_reference_cubes(self, fmt_out='ENVI', try_read_dumped_clf=True, sam_classassignment=False,
                                 max_distance='80%', max_angle=6, nmin_unique_spectra=50, progress=True):
        # type: (str, bool, bool, int, Union[int, float, str], Union[int, float, str], bool) -> ReferenceCube_Generator.refcubes  # noqa
        """Generate reference spectra from all hyperspectral input images.

        Workflow:
        1. Clustering/classification of hyperspectral images and selection of a given number of random signatures
            (a. Spectral downsamling to lower spectral resolution (speedup))
            b. KMeans clustering
            c. Selection of the same number of signatures from each cluster to avoid unequal amount of training data.
        2. Spectral resampling of the selected hyperspectral signatures (for each input image)
        3. Add resampled spectra to reference cubes for each target sensor and write cubes to disk

        :param fmt_out:             output format (GDAL driver code)
        :param try_read_dumped_clf: try to read a prediciouly serialized KMeans classifier from disk
                                    (massively speeds up the RefCube generation)
        :param sam_classassignment: False: use minimal euclidian distance to assign classes to cluster centers
                                    True: use the minimal spectral angle to assign classes to cluster centers
        :param max_distance:    spectra with a larger spectral distance than the given value will be excluded from
                                random sampling.
                                - if given as string like '20%', the maximum spectral distance is computed as 20%
                                  percentile within each cluster
        :param max_angle:       spectra with a larger spectral angle than the given value will be excluded from
                                random sampling.
                                - if given as string like '20%', the maximum spectral angle is computed as 20%
                                  percentile within each cluster
        :param nmin_unique_spectra:   in case a cluster has less than the given number,
                                      do not include it in the reference cube (default: 50)
        :param progress:            show progress bar (default: True)
        :return:                    np.array: [tgt_n_samples x images x spectral bands of the target sensor]
        """
        for im in self.ims_ref:  # type: Union[str, GeoArray]
            # TODO implement check if current image is already included in the refcube -> skip in that case
            src_im = GeoArray(im)

            # get random spectra of the original (hyperspectral) image, equally distributed over all computed clusters
            baseN = os.path.splitext(os.path.basename(im))[0] if isinstance(im, str) else im.basename
            unif_random_spectra = \
                self.cluster_image_and_get_uniform_spectra(src_im,
                                                           try_read_dumped_clf=try_read_dumped_clf,
                                                           sam_classassignment=sam_classassignment,
                                                           max_distance=max_distance,
                                                           max_angle=max_angle,
                                                           nmin_unique_spectra=nmin_unique_spectra,
                                                           progress=progress,
                                                           basename_clf_dump=baseN).astype(np.int16)

            # resample the set of random spectra to match the spectral characteristics of all target sensors
            for tgt_sat, tgt_sen in self.tgt_sat_sen_list:
                # perform spectral resampling
                self.logger.info('Performing spectral resampling to match %s %s specifications...' % (tgt_sat, tgt_sen))
                unif_random_spectra_rsp = self.resample_spectra(
                    unif_random_spectra,
                    src_cwl=np.array(src_im.meta.band_meta['wavelength'], dtype=np.float).flatten(),
                    tgt_srf=self._get_tgt_SRF_object(tgt_sat, tgt_sen),
                    nodataVal=src_im.nodata)

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

    def cluster_image_and_get_uniform_spectra(self, im, downsamp_sat=None, downsamp_sen=None, basename_clf_dump='',
                                              try_read_dumped_clf=True, sam_classassignment=False, max_distance='80%',
                                              max_angle=6, nmin_unique_spectra=50, progress=False):
        # type: (Union[str, GeoArray, np.ndarray], str, str, str, bool, bool, int, Union[int, float, str], Union[int, float, str], bool) -> np.ndarray  # noqa
        """Compute KMeans clusters for the given image and return the an array of uniform random samples.

        :param im:              image to be clustered
        :param downsamp_sat:    satellite code used for intermediate image dimensionality reduction (input image is
                                spectrally resampled to this satellite before it is clustered). requires downsamp_sen.
                                If it is None, no intermediate downsampling is performed.
        :param downsamp_sen:    sensor code used for intermediate image dimensionality reduction (requires downsamp_sat)
        :param basename_clf_dump:   basename of serialized KMeans classifier
        :param try_read_dumped_clf: try to read a previously serialized KMeans classifier from disk
                                    (massively speeds up the RefCube generation)
        :param sam_classassignment: False: use minimal euclidian distance to assign classes to cluster centers
                                    True: use the minimal spectral angle to assign classes to cluster centers
        :param max_distance:    spectra with a larger spectral distance than the given value will be excluded from
                                random sampling.
                                - if given as string like '20%', the maximum spectral distance is computed as 20%
                                  percentile within each cluster
        :param max_angle:       spectra with a larger spectral angle than the given value will be excluded from
                                random sampling.
                                - if given as string like '20%', the maximum spectral angle is computed as 20%
                                  percentile within each cluster
        :param nmin_unique_spectra:   in case a cluster has less than the given number,
                                      do not include it in the reference cube (default: 50)
        :param progress:        whether to show progress bars or not
        :return:    2D array (rows: tgt_n_samples, columns: spectral information / bands
        """
        # input validation
        if downsamp_sat and not downsamp_sen or downsamp_sen and not downsamp_sat:
            raise ValueError("The parameters 'spec_rsp_sat' and 'spec_rsp_sen' must both be provided or completely "
                             "omitted.")

        im2clust = GeoArray(im)

        # get a KMeans classifier for the hyperspectral image
        path_clf = os.path.join(self.dir_clf_dump, '%s__KMeansClf.dill' % basename_clf_dump)
        if not sam_classassignment:
            path_clustermap = os.path.join(self.dir_clf_dump,
                                           '%s__KMeansClusterMap_nclust%d.bsq' % (basename_clf_dump, self.n_clusters))
        else:
            path_clustermap = os.path.join(self.dir_clf_dump,
                                           '%s__SAMClusterMap_nclust%d.bsq' % (basename_clf_dump, self.n_clusters))

        if try_read_dumped_clf and os.path.isfile(path_clf):
            # read the previously dumped classifier from disk
            self.logger.info('Reading previously serialized KMeans classifier from %s...' % path_clf)
            kmeans = KMeansRSImage.from_disk(path_clf=path_clf, im=im2clust)

        else:
            # create KMeans classifier
            # first, perform spectral resampling to Sentinel-2 to reduce dimensionality (speedup)
            if downsamp_sat and downsamp_sen:
                # NOTE: The KMeansRSImage class already reduces the input data to 1 million spectra by default
                from gms_preprocessing.model.gms_object import GMS_identifier
                from gms_preprocessing.io.input_reader import SRF
                tgt_srf = SRF(GMS_identifier(satellite=downsamp_sat, sensor=downsamp_sen, subsystem=None,
                                             image_type='RSD', dataset_ID=-9999, proc_level='L1A', logger=self.logger))
                im2clust = self.resample_image_spectrally(im2clust, tgt_srf, progress=progress)  # output = int16

            # compute KMeans clusters for the spectrally resampled image
            # NOTE: Nodata values are ignored during KMeans clustering.
            self.logger.info('Computing %s KMeans clusters from the input image %s...'
                             % (self.n_clusters, im2clust.basename))
            kmeans = KMeansRSImage(im2clust, n_clusters=self.n_clusters, sam_classassignment=sam_classassignment,
                                   CPUs=self.CPUs, v=self.v)
            kmeans.compute_clusters()
            kmeans.compute_spectral_distances()

            if self.dir_clf_dump and basename_clf_dump:
                kmeans.save_clustermap(path_clustermap)
                kmeans.dump(path_clf)

        if self.v:
            kmeans.plot_cluster_centers()
            kmeans.plot_cluster_histogram()

        # randomly grab the given number of spectra from each cluster, restricted to the 30 % purest spectra
        #   -> no spectra containing nodata values are returned
        self.logger.info('Getting %s random spectra from each cluster...' % (self.tgt_n_samples // self.n_clusters))
        random_samples = kmeans.get_random_spectra_from_each_cluster(samplesize=self.tgt_n_samples // self.n_clusters,
                                                                     max_distance=max_distance,
                                                                     max_angle=max_angle,
                                                                     nmin_unique_spectra=nmin_unique_spectra)
        # random_samples = kmeans\
        #     .get_purest_spectra_from_each_cluster(src_im=GeoArray(im),
        #                                           samplesize=self.tgt_n_samples // self.n_clusters)

        # combine the spectra (2D arrays) of all clusters to a single 2D array
        self.logger.info('Combining random samples from all clusters.')
        random_samples = np.vstack([random_samples[clusterlabel] for clusterlabel in random_samples])

        return random_samples

    def resample_spectra(self, spectra, src_cwl, tgt_srf, nodataVal):
        # type: (Union[GeoArray, np.ndarray], Union[list, np.array], SRF, int) -> np.ndarray
        """Perform spectral resampling of the given image to match the given spectral response functions.

        :param spectra:     2D array (rows: spectral samples;  columns: spectral information / bands
        :param src_cwl:     central wavelength positions of input spectra
        :param tgt_srf:     target spectral response functions to be used for spectral resampling
        :param nodataVal:   nodata value of the given spectra to be ignored during resampling
        :return:
        """
        spectra = GeoArray(spectra)

        # perform spectral resampling of input image to match spectral properties of target sensor
        self.logger.info('Performing spectral resampling to match spectral properties of %s %s...'
                         % (tgt_srf.satellite, tgt_srf.sensor))

        SR = SpectralResampler(src_cwl, tgt_srf)
        spectra_rsp = SR.resample_spectra(spectra,
                                          chunksize=200,
                                          nodataVal=nodataVal,
                                          alg_nodata='radical',
                                          CPUs=self.CPUs)

        return spectra_rsp

    def resample_image_spectrally(self, src_im, tgt_srf, src_nodata=None, progress=False):
        # type: (Union[str, GeoArray], SRF, Union[float, int], bool) -> Union[GeoArray, None]
        """Perform spectral resampling of the given image to match the given spectral response functions.

        :param src_im:      source image to be resampled
        :param tgt_srf:     target spectral response functions to be used for spectral resampling
        :param src_nodata:  source image nodata value
        :param progress:    show progress bar (default: false)
        :return:
        """
        # handle src_im provided as file path or GeoArray instance
        if isinstance(src_im, str):
            im_name = os.path.basename(src_im)
            im_gA = GeoArray(src_im, nodata=src_nodata)
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

        tgt_im = GeoArray(np.zeros((*im_gA.shape[:2], len(tgt_srf.bands)), dtype=np.int16),
                          geotransform=im_gA.gt,
                          projection=im_gA.prj,
                          nodata=im_gA.nodata)
        tgt_im.meta.band_meta['wavelength'] = list(tgt_srf.wvl)
        tgt_im.bandnames = ['B%s' % i if len(i) == 2 else 'B0%s' % i for i in tgt_srf.LayerBandsAssignment]

        tiles = im_gA.tiles((1000, 1000))  # use tiles to save memory
        for ((rS, rE), (cS, cE)), tiledata in (tqdm(tiles) if progress else tiles):
            tgt_im[rS: rE + 1, cS: cE + 1, :] = SR.resample_image(tiledata.astype(np.int16),
                                                                  nodataVal=im_gA.nodata,
                                                                  alg_nodata='radical',
                                                                  CPUs=self.CPUs)

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
        self.logger = logger or SpecHomo_Logger(__name__)  # must be pickable

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
        # get L1C bands (after AC)
        # NOTE: - signal of additional bands at L1A is not predictable by spectral harmonization because these bands
        #         are not driven by surface albedo but by atmospheric conditions (945, 1373 nm)
        #       - Landsat-8 band 9 (PAN) is currently not supported by AC => L1C_withPan does not exist  # FIXME
        from gms_preprocessing.model.gms_object import GMS_identifier
        from gms_preprocessing.model.metadata import get_LayerBandsAssignment
        L1C_GMSid = GMS_identifier(satellite=satellite, sensor=sensor, subsystem=None, image_type='RSD',
                                   dataset_ID=-9999, proc_level='L1C', logger=None)

        LBAs = []
        for lba in [get_LayerBandsAssignment(L1C_GMSid, no_pan=False, sort_by_cwl=True),  # L1C_withPan_cwlSorted
                    get_LayerBandsAssignment(L1C_GMSid, no_pan=True, sort_by_cwl=True),  # L1C_noPan_cwlSorted
                    get_LayerBandsAssignment(L1C_GMSid, no_pan=False, sort_by_cwl=False),  # L1C_withPan_alphabetical
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
                # noinspection PyProtectedMember
                setattr(ML._final_estimator, attr, getattr(ML._final_estimator, attr).astype(np.float32))
            else:
                try:
                    setattr(ML, attr, getattr(ML, attr).astype(np.float32))
                except AttributeError:
                    pass

        return ML

    def create_classifiers(self, outDir, method='LR', n_clusters=50, sam_classassignment=False, CPUs=24,
                           max_distance=options['classifiers']['trainspec_filtering']['max_distance'],
                           max_angle=options['classifiers']['trainspec_filtering']['max_angle'],
                           **kwargs):
        # type: (str, str, int, bool, int, int, int, dict) -> None
        """Create cluster classifiers for all combinations of the reference cubes given in __init__().

        :param outDir:      output directory for the created cluster classifier collections
        :param method:      type of machine learning classifiers to be included in classifier collections
                            'LR':   Linear Regression
                            'RR':   Ridge Regression
                            'QR':   Quadratic Regression
                            'RFR':  Random Forest Regression (50 trees with maximum depth of 3 by default)
        :param n_clusters:  number of clusters to be used for KMeans clustering
        :param sam_classassignment: False: use minimal euclidian distance to assign classes to cluster centers
                                    True: use the minimal spectral angle to assign classes to cluster centers
        :param CPUs:        number of CPUs to be used for KMeans clustering
        :param max_distance:    maximum spectral distance allowed during filtering of training spectra
                                - if given as string, e.g., '80%' means that the worst 20 % of the input spectra are
                                  excluded)
        :param max_angle:       maximum spectral angle allowed during filtering of training spectra
                                 - if given as string, e.g., '80%' means that the worst 20 % of the input spectra are
                                   excluded)
        :param kwargs:      keyword arguments to be passed to machine learner
        """
        # validate and set defaults
        if method == 'RFR':
            if n_clusters > 1:
                self.logger.warning("The spectral homogenization method 'Random Forest Regression' does not allow "
                                    "spectral sub-clustering. Setting 'n_clusters' to 1.")
                n_clusters = 1

            kwargs.update(dict(
                n_jobs=kwargs.get('n_jobs', CPUs),
                n_estimators=kwargs.get('n_estimators', options['classifiers']['RFR']['n_trees']),
                max_depth=kwargs.get('max_depth', options['classifiers']['RFR']['max_depth'])
            ))

        # build the classifier collections with separate classifiers for each cluster
        for src_cube in self.refcubes:  # type: RefCube
            cls_collection = nested_dict()

            # get cluster labels for each source cube separately (as they depend on source spectral characteristics)
            clusterlabels_src_cube, spectral_distances, spectral_angles = \
                self._get_cluster_labels_for_source_refcube(src_cube, n_clusters, CPUs,
                                                            sam_classassignment=sam_classassignment,
                                                            return_spectral_distances=True, return_spectral_angles=True)

            for tgt_cube in self.refcubes:
                if (src_cube.satellite, src_cube.sensor) == (tgt_cube.satellite, tgt_cube.sensor):
                    continue
                self.logger.info("Creating %s cluster classifier to predict %s %s from %s %s..."
                                 % (method, tgt_cube.satellite, tgt_cube.sensor, src_cube.satellite, src_cube.sensor))

                src_derived_LBAs = self._get_derived_LayerBandsAssignments(src_cube.satellite, src_cube.sensor)
                tgt_derived_LBAs = self._get_derived_LayerBandsAssignments(tgt_cube.satellite, tgt_cube.sensor)

                for src_LBA, tgt_LBA in product(src_derived_LBAs, tgt_derived_LBAs):
                    self.logger.debug('Creating %s cluster classifier for LBA %s => %s...'
                                      % (method, '_'.join(src_LBA), '_'.join(tgt_LBA)))

                    # Get center wavelength positions
                    # NOTE: they cannot be taken from RefCube instances because they always represent L1C LBAs
                    from gms_preprocessing.model.metadata import get_center_wavelengths_by_LBA
                    src_wavelengths = get_center_wavelengths_by_LBA(src_cube.satellite, src_cube.sensor, src_LBA)
                    tgt_wavelengths = get_center_wavelengths_by_LBA(tgt_cube.satellite, tgt_cube.sensor, tgt_LBA)

                    # Get training data for source and target image according to the given LayerBandsAssignments
                    # e.g., source: Landsat 7 image in LBA 1__2__3__4__5__7 and target L8 in 1__2__3__4__5__6__7
                    df_src_spectra_allclust = src_cube.get_spectra_dataframe(src_LBA)
                    df_tgt_spectra_allclust = tgt_cube.get_spectra_dataframe(tgt_LBA)

                    # assign clusterlabel and spectral distance to the cluster center to each spectrum
                    for df in [df_src_spectra_allclust, df_tgt_spectra_allclust]:
                        df.insert(0, 'cluster_label', clusterlabels_src_cube)
                        df.insert(1, 'spectral_distance', spectral_distances)
                        df.insert(2, 'spectral_angle', spectral_angles)

                    # remove spectra with cluster label -9999
                    # (clusters with too few spectra that are set to nodata in the refcube)
                    df_src_spectra_allclust = df_src_spectra_allclust[df_src_spectra_allclust.cluster_label != -9999]
                    df_tgt_spectra_allclust = df_tgt_spectra_allclust[df_tgt_spectra_allclust.cluster_label != -9999]

                    # ensure source and target spectra do not contain nodata values (would affect classifiers)
                    assert src_cube.data.nodata is None or src_cube.data.nodata not in df_src_spectra_allclust.values
                    assert tgt_cube.data.nodata is None or tgt_cube.data.nodata not in df_tgt_spectra_allclust.values

                    for clusterlabel in range(n_clusters):
                        self.logger.debug('Creating %s classifier for cluster %s...' % (method, clusterlabel))

                        df_src_spectra_best, df_tgt_spectra_best = \
                            self._extract_best_spectra_from_cluster(
                                clusterlabel, df_src_spectra_allclust, df_tgt_spectra_allclust,
                                max_distance=max_distance, max_angle=max_angle)

                        # Set train and test variables for the classifier
                        src_spectra_curlabel = df_src_spectra_best.values[:, 3:]
                        tgt_spectra_curlabel = df_tgt_spectra_best.values[:, 3:]

                        train_src, test_src, train_tgt, test_tgt = \
                            train_test_split(src_spectra_curlabel, tgt_spectra_curlabel,
                                             test_size=0.4, shuffle=True, random_state=0)

                        # train the learner and add metadata
                        ML = self.train_machine_learner(train_src, train_tgt, test_src, test_tgt, method, **kwargs)
                        # noinspection PyTypeChecker
                        ML = self._add_metadata_to_machine_learner(
                            ML, src_cube, tgt_cube, src_LBA, tgt_LBA, src_wavelengths, tgt_wavelengths,
                            train_src, train_tgt, n_clusters, clusterlabel)

                        # append to classifier collection
                        cls_collection[
                            '__'.join(src_LBA)][tgt_cube.satellite, tgt_cube.sensor][
                            '__'.join(tgt_LBA)][clusterlabel] = ML

            # dump to disk
            fName_cls = get_filename_classifier_collection(method, src_cube.satellite, src_cube.sensor,
                                                           n_clusters=n_clusters, **kwargs)

            with open(os.path.join(outDir, fName_cls), 'wb') as outF:
                dill.dump(cls_collection.to_dict(), outF, protocol=dill.HIGHEST_PROTOCOL)

    def _get_cluster_labels_for_source_refcube(self, src_cube, n_clusters, CPUs, sam_classassignment=False,
                                               return_spectral_distances=False, return_spectral_angles=False):
        """Get cluster labels for each source cube separately

        NOTE: - We use the GMS L1C bands (without atmospheric bands and PAN-band) for clustering.
              - clustering is only performed on the source cube because the source sensor spectral information
                is later used to assign the correct homogenization classifier to each pixel

        :param src_cube:
        :param n_clusters:
        :param sam_classassignment:         False: use minimal euclidian distance to assign classes to cluster centers
                                            True: use the minimal spectral angle to assign classes to cluster centers
        :param CPUs:
        :param return_spectral_distances:
        :return:
        """
        self.logger.info('Clustering %s %s reference cube (%s clusters)...'
                         % (src_cube.satellite, src_cube.sensor, n_clusters))
        from gms_preprocessing.model.gms_object import GMS_identifier
        from gms_preprocessing.model.metadata import get_LayerBandsAssignment
        L1C_GMSid = GMS_identifier(satellite=src_cube.satellite, sensor=src_cube.sensor, subsystem=None,
                                   image_type='RSD', dataset_ID=-9999, proc_level='L1C', logger=None)
        LBA2clust = get_LayerBandsAssignment(L1C_GMSid, no_pan=True, sort_by_cwl=True)
        src_data2clust = src_cube.get_band_combination(LBA2clust)
        km = KMeansRSImage(src_data2clust, n_clusters=n_clusters, sam_classassignment=sam_classassignment, CPUs=CPUs)
        km.compute_clusters()

        return_vals = [km.labels_with_nodata]

        if return_spectral_distances:
            return_vals.append(km.spectral_distances_with_nodata)
        if return_spectral_angles:
            return_vals.append(km.spectral_angles_with_nodata)

        return tuple(return_vals)

    @staticmethod
    def _extract_best_spectra_from_cluster(clusterlabel, df_src_spectra_allclust, df_tgt_spectra_allclust,
                                           max_distance, max_angle):
        # NOTE: We exclude the noisy spectra with the largest spectral distances to their cluster
        #       center here (random spectra from within the upper 40 %)
        assert len(df_src_spectra_allclust.index) == len(df_tgt_spectra_allclust.index), \
            'Source and target spectra dataframes must have the same number of spectral samples.'

        df_src_spectra = df_src_spectra_allclust[df_src_spectra_allclust.cluster_label == clusterlabel]

        # max_dist = np.percentile(df_src_spectra.spectral_distance, max_distance)
        # max_angle = np.percentile(df_src_spectra.spectral_angle, max_angle_percent)

        if isinstance(max_angle, str):
            max_angle = np.percentile(df_src_spectra.spectral_angle,
                                      int(max_angle.split('%')[0].strip()))

        tmp = df_src_spectra[df_src_spectra.spectral_angle < max_angle]
        if len(tmp.index) > 10:
            df_src_spectra = tmp

        if isinstance(max_distance, str):
            max_distance = np.percentile(df_src_spectra.spectral_distance,
                                         int(max_distance.split('%')[0].strip()))

        tmp = df_src_spectra[df_src_spectra.spectral_distance < max_distance]
        if len(tmp.index) > 10:
            df_src_spectra = tmp

        if len(df_src_spectra.index) > 1700:
            df_src_spectra = df_src_spectra.sort_values(by='spectral_angle').head(1700)

        df_tgt_spectra = df_tgt_spectra_allclust.loc[df_src_spectra.index, :]

        return df_src_spectra, df_tgt_spectra

    @staticmethod
    def _add_metadata_to_machine_learner(ML, src_cube, tgt_cube, src_LBA, tgt_LBA, src_wavelengths, tgt_wavelengths,
                                         src_train_spectra, tgt_train_spectra, n_clusters, clusterlabel):
        # compute cluster center for source spectra (only on those spectra used for model training)
        cluster_center = np.mean(src_train_spectra, axis=0)
        cluster_median = np.median(src_train_spectra, axis=0)
        sample_spectra = DataFrame(src_train_spectra).sample(100, replace=True, random_state=20).values

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
        ML.cluster_center = cluster_center.astype(src_cube.data.dtype)
        ML.cluster_median = cluster_median.astype(src_cube.data.dtype)
        ML.cluster_sample_spectra = sample_spectra.astype(np.int16)  # scaled between 0 and 10000
        ML.tgt_cluster_center = np.mean(tgt_train_spectra, axis=0).astype(tgt_cube.data.dtype)
        ML.tgt_cluster_median = np.median(tgt_train_spectra, axis=0).astype(tgt_cube.data.dtype)

        assert len(ML.src_LBA) == len(ML.src_wavelengths)
        assert len(ML.tgt_LBA) == len(ML.tgt_wavelengths)

        return ML


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


def get_filename_classifier_collection(method, src_satellite, src_sensor, n_clusters=1, **cls_kwinit):
    if method == 'RR':
        method += '_alpha1.0'  # TODO add to config
    elif method == 'RFR':
        # assert 'n_estimators' in cls_kwinit
        # method += '_trees%s' % cls_kwinit['n_estimators']  # we statically use 50 trees for Random Forest Regression
        method += '_trees50'  # FIXME   # we statically use 50 trees for Random Forest Regression

    return '__'.join(['%s_clust%s' % (method, n_clusters), src_satellite, src_sensor]) + '.dill'


# def get_classifier_filename(method, src_satellite, src_sensor, src_LBA_name, tgt_satellite, tgt_sensor, tgt_LBA_name):
#     return '__'.join([method, src_satellite, src_sensor, src_LBA_name]) + \
#                     '__to__' + '__'.join([tgt_satellite, tgt_sensor, tgt_LBA_name]) + '.dill'
