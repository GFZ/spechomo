# -*- coding: utf-8 -*-
import os
import tempfile
import zipfile
from collections import OrderedDict
from pprint import pformat
from typing import Union, List  # noqa F401  # flake8 issue

from tqdm import tqdm
import dill
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from geoarray import GeoArray  # noqa F401  # flake8 issue

from .classifier_creation import get_filename_classifier_collection, get_machine_learner
from .exceptions import ClassifierNotAvailableError
from .utils import im2spectra, spectra2im


class Cluster_Learner(object):
    """
    A class that holds the machine learning classifiers for multiple spectral clusters as well as a global classifier.
    These classifiers can be applied to an input sensor image by using the predict method.
    """
    def __init__(self, dict_clust_MLinstances, global_classifier):
        # type: (Union[dict, ClassifierCollection], any) -> None
        """Get an instance of Cluster_Learner.

        :param dict_clust_MLinstances:  a dictionary of cluster specific machine learning classifiers
        :param global_classifier:       the global machine learning classifier to be applied at image positions with
                                        high spectral dissimilarity to the available cluster centers
        """
        self.cluster_pixVals = list(sorted(dict_clust_MLinstances.keys()))  # type: List[int]
        self.MLdict = OrderedDict((clust, dict_clust_MLinstances[clust]) for clust in self.cluster_pixVals)
        self.global_clf = global_classifier
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

    @classmethod
    def from_disk(cls, classifier_rootDir, method, n_clusters,
                  src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA):
        # type: (str, str, int, str, str, list, str, str, list) -> Cluster_Learner
        """Read a previously saved ClusterLearner from disk and return a ClusterLearner instance.

        Describe the classifier specifications with the given arguments.

        :param classifier_rootDir:  root directory of the classifiers
        :param method:              harmonization method
                                    'LR':   Linear Regression
                                    'RR':   Ridge Regression
                                    'QR':   Quadratic Regression
                                    'RFR':  Random Forest Regression  (50 trees; does not allow spectral sub-clustering)
        :param n_clusters:          number of clusters
        :param src_satellite:       source satellite, e.g., 'Landsat-8'
        :param src_sensor:          source sensor, e.g., 'OLI_TIRS'
        :param src_LBA:             source LayerBandsAssignment
        :param tgt_satellite:       target satellite, e.g., 'Landsat-8'
        :param tgt_sensor:          target sensor, e.g., 'OLI_TIRS'
        :param tgt_LBA:             target LayerBandsAssignment
        :return:                    classifier instance loaded from disk
        """
        # get path of classifier zip archive
        path_classifier_zip = os.path.join(classifier_rootDir, '%s_classifiers.zip' % method)
        args = (method, src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA)

        if os.path.isfile(path_classifier_zip):
            # get cluster specific classifiers and store them in a ClassifierCollection dictionary
            dict_clust_MLinstances = cls._read_ClassifierCollection_from_zipFile(
                path_classifier_zip, *args, n_clusters=n_clusters)

            # get the global classifier and add it as cluster label '-1'
            global_clf = cls._read_ClassifierCollection_from_zipFile(
                path_classifier_zip, *args, n_clusters=1)[0]

        elif os.path.isdir(classifier_rootDir):
            # get cluster specific classifiers and store them in a ClassifierCollection dictionary
            fName_clf_clustN = get_filename_classifier_collection(method, src_satellite, src_sensor,
                                                                  n_clusters=n_clusters)
            dict_clust_MLinstances = cls._read_ClassifierCollection_from_unzippedFile(
                os.path.join(classifier_rootDir, fName_clf_clustN), *args, n_clusters=n_clusters)

            # get the global classifier and add it as cluster label '-1'
            fName_clf_clust1 = get_filename_classifier_collection(method, src_satellite, src_sensor, n_clusters=1)
            global_clf = cls._read_ClassifierCollection_from_unzippedFile(
                os.path.join(classifier_rootDir, fName_clf_clust1), *args, n_clusters=1)[0]

        else:
            raise FileNotFoundError("No '%s' classifiers available at %s." % (method, path_classifier_zip))

        # create an instance of ClusterLearner based on the ClassifierCollection dictionary
        return Cluster_Learner(dict_clust_MLinstances, global_clf)

    @staticmethod
    def _read_ClassifierCollection_from_zipFile(path_classifier_zip, method, src_satellite, src_sensor,
                                                src_LBA, tgt_satellite, tgt_sensor, tgt_LBA, n_clusters):
        # type: (str, str, str, str, list, str, str, list, int) -> ClassifierCollection

        # read requested classifier from zip archive and create a ClassifierCollection
        with zipfile.ZipFile(path_classifier_zip, "r") as zf, tempfile.TemporaryDirectory() as td:
            fName_clf = get_filename_classifier_collection(method, src_satellite, src_sensor, n_clusters=n_clusters)
            zf.extract(fName_clf, td)
            path_clf = os.path.join(td, fName_clf)

            return Cluster_Learner._read_ClassifierCollection_from_unzippedFile(
                path_clf, method, src_satellite, src_sensor, src_LBA,
                tgt_satellite, tgt_sensor, tgt_LBA, n_clusters)

    @staticmethod
    def _read_ClassifierCollection_from_unzippedFile(path_classifier, method, src_satellite, src_sensor, src_LBA,
                                                     tgt_satellite, tgt_sensor, tgt_LBA, n_clusters):
        # type: (str, str, str, str, list, str, str, list, int) -> ClassifierCollection

        # read requested classifier from zip archive and create a ClassifierCollection
        try:
            clf_collection = \
                ClassifierCollection(path_classifier)['__'.join(src_LBA)][tgt_satellite, tgt_sensor]['__'.join(tgt_LBA)]
        except KeyError:
            raise ClassifierNotAvailableError(method, src_satellite, src_sensor, src_LBA,
                                              tgt_satellite, tgt_sensor, tgt_LBA, n_clusters)

        # validation
        expected_MLtype = type(get_machine_learner(method))
        if len(clf_collection.keys()) != n_clusters:
            raise RuntimeError('Read classifier with %s clusters instead of %s.'
                               % (len(clf_collection.keys()), n_clusters))
        for label, ml in clf_collection.items():
            if not isinstance(ml, expected_MLtype):
                raise ValueError("The given dillFile %s contains a spectral cluster (label '%s') with a %s machine "
                                 "learner instead of the expected %s."
                                 % (os.path.basename(path_classifier), label, type(ml), expected_MLtype.__name__,))

        return clf_collection

    def __iter__(self):
        for cluster in self.cluster_pixVals:
            yield self.MLdict[cluster]

    def predict(self, im_src, cmap, nodataVal=None, cmap_nodataVal=None, cmap_unclassifiedVal=-1):
        # type: (Union[np.ndarray, GeoArray], np.ndarray, Union[int, float], Union[int, float], Union[int, float]) -> np.ndarray  # noqa
        """Predict target satellite spectral information using separate prediction coefficients for spectral clusters.

        :param im_src:          input image to be used for prediction
        :param cmap:            classification map that assigns each image spectrum to a corresponding cluster
                                -> must be a 2D np.ndarray with the same X-/Y-dimension like im_src
        :param nodataVal:       nodata value to be used to fill into the predicted image
        :param cmap_nodataVal:  nodata class value of the nodata class of the classification map
        :param cmap_unclassifiedVal:    'unclassified' class value of the nodata class of the classification map
        :return:
        """
        cluster_labels = sorted(list(np.unique(cmap)))

        im_pred = np.full((im_src.shape[0], im_src.shape[1], self.tgt_n_bands),
                          fill_value=nodataVal if nodataVal is not None else 0,
                          dtype=im_src.dtype)
        out_dTMin, out_dTMax = np.iinfo(im_src.dtype).min, np.iinfo(im_src.dtype).max

        if len(cluster_labels) > 1:
            # iterate over all cluster labels and apply corresponding machine learner parameters
            # to predict target spectra

            for pixVal in cluster_labels:
                if pixVal == cmap_nodataVal:
                    continue

                elif pixVal == cmap_unclassifiedVal:
                    # apply global homogenization coefficients
                    classifier = self.global_clf

                else:
                    # apply cluster specific homogenization coefficients
                    classifier = self.MLdict[pixVal]

                mask_pixVal = cmap == pixVal
                spectra_pred = classifier.predict(im_src[mask_pixVal])

                if spectra_pred.min() >= out_dTMin and spectra_pred.max() <= out_dTMax:
                    im_pred[mask_pixVal] = spectra_pred.astype(im_src.dtype)
                else:
                    raise TypeError('Predicted values for class %d exceed the value range of the output data type '
                                    '(%s - %s).' % (pixVal, np.iinfo(im_src.dtype.min), np.iinfo(im_src.dtype.max)))

        else:
            # predict target spectra directly (much faster than the above algorithm)
            pixVal = cluster_labels[0]

            if pixVal == cmap_nodataVal:
                # im_src consists only of no data values
                pass  # im_pred keeps at nodataVal

            else:
                if pixVal == cmap_unclassifiedVal:
                    # apply global homogenization coefficients
                    classifier = self.global_clf
                else:
                    classifier = self.MLdict[pixVal]
                    assert classifier.clusterlabel == pixVal

                spectra = im2spectra(im_src)
                spectra_pred = classifier.predict(spectra)

                if spectra_pred.min() >= out_dTMin and spectra_pred.max() <= out_dTMax:
                    im_pred = spectra2im(spectra_pred.astype(im_src.dtype), im_src.shape[0], im_src.shape[1])
                else:
                    raise TypeError('Predicted values for class %d exceed the value range of the output data type '
                                    '(%s - %s).' % (pixVal, np.iinfo(im_src.dtype.min), np.iinfo(im_src.dtype.max)))

        return im_pred

    def predict_weighted_averages(self, im_src, cmap_3D, weights_3D=None, nodataVal=None,
                                  cmap_nodataVal=None, cmap_unclassifiedVal=-1):
        # type: (Union[np.ndarray, GeoArray], np.ndarray, np.ndarray, Union[int, float], Union[int, float], Union[int, float]) -> np.ndarray  # noqa
        """Predict target satellite spectral information using separate prediction coefficients for spectral clusters.

        NOTE:   This version of the prediction function uses the prediction coefficients of multiple spectral clusters
                and computes the result as weighted average of them. Therefore, the classifcation map must assign
                multiple spectral cluster to each input pixel.
        NOTE:   At unclassified pixel positions (cmap_3D[y,x,z>0] == -1) the prediction result using global coefficients
                is ignored in the weighted average. In that case the prediction result is based on the found valid
                spectral clusters and is not affected by the global coefficients (should improve prediction results).

        :param im_src:          input image to be used for prediction
        :param cmap_3D:         classification map that assigns each image spectrum to multiple corresponding clusters
                                -> must be a 3D np.ndarray with the same X-/Y-dimension like im_src
        :param weights_3D:
        :param nodataVal:       nodata value to be used to fill into the predicted image
        :param cmap_nodataVal:  nodata class value of the nodata class of the classification map
        :param cmap_unclassifiedVal:    'unclassified' class value of the nodata class of the classification map
        :return:
        """
        if not cmap_3D.ndim > 2:
            raise ValueError('Input classification map needs at least 2 bands to compute prediction results as'
                             'weighted averages.')

        if cmap_3D.shape != weights_3D.shape:
            raise ValueError("The input arrays 'cmap_3D' and 'weights_3D' need to have the same dimensions. "
                             "Received %s vs. %s." % (cmap_3D.shape, weights_3D.shape))

        # predict for each classification map band
        ims_pred_temp = []

        for band in range(cmap_3D.shape[2]):
            ims_pred_temp.append(
                self.predict(im_src, cmap_3D[:, :, band],
                             nodataVal=nodataVal,
                             cmap_nodataVal=cmap_nodataVal,
                             cmap_unclassifiedVal=cmap_unclassifiedVal
                             ))

        # merge classification results by weighted averaging
        nsamp, nbandpred, nbandscmap = np.dot(*weights_3D.shape[:2]), ims_pred_temp[0].shape[2], weights_3D.shape[2]
        weights = \
            np.ones((nsamp, nbandpred, nbandscmap)) if weights_3D is None else \
            np.tile(weights_3D.reshape(nsamp, 1, nbandscmap), (1, nbandpred, 1))

        # set weighting of unclassified pixel positions to zero (except from the first cmap band)
        #   -> see NOTE #2 in the docstring
        mask_unclassif = np.tile(cmap_3D.reshape(nsamp, 1, nbandscmap), (1, nbandpred, 1)) == cmap_unclassifiedVal
        mask_unclassif[:, :, 0] = False  # if all other clusters are invalid, at least the first one is used for prediction # noqa
        weights[mask_unclassif] = 0

        spectra_pred = np.average(np.dstack([im2spectra(im) for im in ims_pred_temp]), weights=weights, axis=2)
        im_pred = spectra2im(spectra_pred, tgt_rows=im_src.shape[0], tgt_cols=im_src.shape[1])

        return im_pred

    def plot_sample_spectra(self, cluster_label='all', include_mean_spectrum=True, include_median_spectrum=True,
                            ncols=5, **kw_fig):
        # type: (Union[str, int, List], bool, bool, int, dict) -> plt.figure

        if isinstance(cluster_label, int):
            lbls2plot = [cluster_label]
        elif isinstance(cluster_label, list):
            lbls2plot = cluster_label
        elif cluster_label == 'all':
            lbls2plot = list(range(self.n_clusters))
        else:
            raise ValueError(cluster_label)

        # create a single plot
        if len(lbls2plot) == 1:
            if cluster_label == 'all':
                cluster_label = 0

            fig, axes = plt.figure(), None
            for i in range(100):
                plt.plot(self.src_wavelengths, self.MLdict[cluster_label].cluster_sample_spectra[i, :])

            plt.xlabel('wavelength [nm]')
            plt.ylabel('%s %s\nreflectance [0-10000]' % (self.src_satellite, self.src_sensor))
            plt.title('Cluster #%s' % cluster_label)
            plt.grid(lw=0.2)
            plt.ylim(0, 10000)

            if include_mean_spectrum:
                plt.plot(self.src_wavelengths, self.MLdict[cluster_label].cluster_center, c='black', lw=3)
            if include_median_spectrum:
                plt.plot(self.src_wavelengths, np.median(self.MLdict[cluster_label].cluster_sample_spectra, axis=0),
                         '--', c='black', lw=3)

        # create a plot with multiple subplots
        else:
            nplots = len(lbls2plot)
            ncols = nplots if nplots < ncols else ncols
            nrows = nplots // ncols if not nplots % ncols else nplots // ncols + 1
            figsize = (4 * ncols, 3 * nrows)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex='all', sharey='all',
                                     **kw_fig)

            for lbl, ax in tqdm(zip(lbls2plot, axes.flatten()), total=nplots):
                for i in range(100):
                    ax.plot(self.src_wavelengths, self.MLdict[lbl].cluster_sample_spectra[i, :], lw=1)

                if include_mean_spectrum:
                    ax.plot(self.src_wavelengths, self.MLdict[lbl].cluster_center, c='black', lw=2)
                if include_median_spectrum:
                    ax.plot(self.src_wavelengths, np.median(self.MLdict[lbl].cluster_sample_spectra, axis=0),
                            '--', c='black', lw=3)

                ax.grid(lw=0.2)
                ax.set_ylim(0, 10000)

                if ax.is_last_row():
                    ax.set_xlabel('wavelength [nm]')
                if ax.is_first_col():
                    ax.set_ylabel('%s %s\nreflectance [0-10000]' % (self.src_satellite, self.src_sensor))
                ax.set_title('Cluster #%s' % lbl)

        plt.tight_layout()
        plt.show()

        return fig, axes

    def _collect_stats(self, cluster_label):
        df = DataFrame(columns=['band', 'wavelength', 'RMSE', 'MAE', 'MAPE'])
        df.band = self.tgt_LBA
        df.wavelength = np.round(self.tgt_wavelengths, 1)
        df.RMSE = np.round(self.MLdict[cluster_label].rmse_per_band, 1)
        df.MAE = np.round(self.MLdict[cluster_label].mae_per_band, 1)
        df.MAPE = np.round(self.MLdict[cluster_label].mape_per_band, 1)

        overall_stats = dict(scores=self.MLdict[cluster_label].scores)

        return df, overall_stats

    def print_stats(self):
        from tabulate import tabulate

        for lbl in range(self.n_clusters):
            print('Cluster #%s:' % lbl)
            band_stats, overall_stats = self._collect_stats(lbl)
            print(overall_stats)
            print(tabulate(band_stats, headers=band_stats.columns))
            print()


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
