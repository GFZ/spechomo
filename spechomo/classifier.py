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

import os
import tempfile
import zipfile
from collections import OrderedDict
from pprint import pformat
from typing import Union, List, TYPE_CHECKING  # noqa F401  # flake8 issue
import json
import builtins

if TYPE_CHECKING:
    from matplotlib import pyplot as plt  # noqa F401  # flake8 issue

from tqdm import tqdm
import dill
import numpy as np
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
                  src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA, n_estimators=50):
        # type: (str, str, int, str, str, list, str, str, list, int) -> Cluster_Learner
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
        :param n_estimators:        number of estimators (only used in case of method=='RFR'
        :return:                    classifier instance loaded from disk
        """
        # get path of classifier zip archive
        args = (method, src_satellite, src_sensor, src_LBA, tgt_satellite, tgt_sensor, tgt_LBA)
        kw_clfinit = dict(n_estimators=n_estimators)

        if os.path.isfile(os.path.join(classifier_rootDir, '%s_classifiers.zip' % method)):
            # get cluster specific classifiers and store them in a ClassifierCollection dictionary
            dict_clust_MLinstances = cls._read_ClassifierCollection_from_zipFile(
                classifier_rootDir, *args, n_clusters=n_clusters, **kw_clfinit)

            # get the global classifier and add it as cluster label '-1'
            global_clf = cls._read_ClassifierCollection_from_zipFile(
                classifier_rootDir, *args, n_clusters=1, **kw_clfinit)[0]

        elif os.path.isdir(classifier_rootDir):
            # get cluster specific classifiers and store them in a ClassifierCollection dictionary
            dict_clust_MLinstances = cls._read_ClassifierCollection_from_unzippedFile(
                classifier_rootDir, *args, n_clusters=n_clusters, **kw_clfinit)

            # get the global classifier and add it as cluster label '-1'
            global_clf = cls._read_ClassifierCollection_from_unzippedFile(
                classifier_rootDir, *args, n_clusters=1, **kw_clfinit)[0]

        else:
            raise FileNotFoundError("No '%s' classifiers available at %s." % (method, classifier_rootDir))

        # create an instance of ClusterLearner based on the ClassifierCollection dictionary
        return Cluster_Learner(dict_clust_MLinstances, global_clf)

    @staticmethod
    def _read_ClassifierCollection_from_zipFile(classifier_rootDir, method, src_satellite, src_sensor,
                                                src_LBA, tgt_satellite, tgt_sensor, tgt_LBA, n_clusters,
                                                **kw_clfinit):
        # type: (str, str, str, str, list, str, str, list, int, dict) -> ClassifierCollection

        path_classifier_zip = os.path.join(classifier_rootDir, '%s_classifiers.zip' % method)

        # read requested classifier from zip archive and create a ClassifierCollection
        with zipfile.ZipFile(path_classifier_zip, "r") as zf, tempfile.TemporaryDirectory() as td:
            fName_clf = get_filename_classifier_collection(method, src_satellite, src_sensor, n_clusters=n_clusters,
                                                           **kw_clfinit)
            try:
                zf.extract(fName_clf, td)
            except KeyError:
                raise FileNotFoundError("No classifiers for %s %s with %d clusters contained in %s."
                                        % (src_satellite, src_sensor, n_clusters, path_classifier_zip))

            return Cluster_Learner._read_ClassifierCollection_from_unzippedFile(
                td, method, src_satellite, src_sensor, src_LBA,
                tgt_satellite, tgt_sensor, tgt_LBA, n_clusters)

    @staticmethod
    def _read_ClassifierCollection_from_unzippedFile(classifier_rootDir, method, src_satellite, src_sensor, src_LBA,
                                                     tgt_satellite, tgt_sensor, tgt_LBA, n_clusters, **kw_clfinit):
        # type: (str, str, str, str, list, str, str, list, int, dict) -> ClassifierCollection

        fName_clf_clustN = get_filename_classifier_collection(method, src_satellite, src_sensor,
                                                              n_clusters=n_clusters, **kw_clfinit)
        path_classifier = os.path.join(classifier_rootDir, fName_clf_clustN)

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
                          dtype=np.float32)

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
                im_pred[mask_pixVal] = classifier.predict(im_src[mask_pixVal])

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
                im_pred = spectra2im(spectra_pred, im_src.shape[0], im_src.shape[1])

        return im_pred  # float32 array

    def predict_weighted_averages(self, im_src, cmap_3D, weights_3D=None, nodataVal=None,
                                  cmap_nodataVal=None, cmap_unclassifiedVal=-1):
        # type: (Union[np.ndarray, GeoArray], np.ndarray, np.ndarray, Union[int, float], Union[int, float], Union[int, float]) -> np.ndarray  # noqa
        """Predict target satellite spectral information using separate prediction coefficients for spectral clusters.

        NOTE:   This version of the prediction function uses the prediction coefficients of multiple spectral clusters
                and computes the result as weighted average of them. Therefore, the classifcation map must assign
                multiple spectral cluster to each input pixel.

        # NOTE:   At unclassified pixels (cmap_3D[y,x,z>0] == -1) the prediction result using global coefficients
        #         is ignored in the weighted average. In that case the prediction result is based on the found valid
        #         spectral clusters and is not affected by the global coefficients (should improve prediction results).

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
            np.tile(weights_3D.reshape(nsamp, 1, nbandscmap), (1, nbandpred, 1))  # nclust x n_tgt_bands x n_cmap_bands

        # set weighting of unclassified pixel positions to zero (except from the first cmap band)
        #   -> see NOTE #2 in the docstring
        # mask_unclassif = np.tile(cmap_3D.reshape(nsamp, 1, nbandscmap), (1, nbandpred, 1)) == cmap_unclassifiedVal
        # mask_unclassif[:, :, :1] = False  # if all other clusters are invalid, at least the first one is used for prediction # noqa
        # weights[mask_unclassif] = 0

        spectra_pred = np.average(np.dstack([im2spectra(im) for im in ims_pred_temp]), weights=weights, axis=2)
        im_pred = spectra2im(spectra_pred, tgt_rows=im_src.shape[0], tgt_cols=im_src.shape[1])

        return im_pred

    def plot_sample_spectra(self, cluster_label='all', include_mean_spectrum=True, include_median_spectrum=True,
                            ncols=5, **kw_fig):
        # type: (Union[str, int, List], bool, bool, int, dict) -> plt.figure
        from matplotlib import pyplot as plt  # noqa

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

    def to_jsonable_dict(self):
        """Create a dictionary containing a JSONable replicate of the current Cluster_Learner instance."""
        common_meta_keys = ['src_satellite', 'src_sensor', 'tgt_satellite', 'tgt_sensor', 'src_LBA', 'tgt_LBA',
                            'src_n_bands', 'tgt_n_bands', 'src_wavelengths', 'tgt_wavelengths', 'n_clusters']
        jsonable_dict = dict()
        decode_types_dict = dict()

        # get jsonable dict for global classifier and add decoding type hints
        jsonable_dict['classifier_global'] =\
            classifier_to_jsonable_dict(self.global_clf, skipkeys=common_meta_keys, include_typesdict=True)
        decode_types_dict['classifiers_all'] = jsonable_dict['classifier_global']['__decode_types']
        del jsonable_dict['classifier_global']['__decode_types']

        # get jsonable dicts for each classifier of self.MLdict and add corresponding decoding type hints
        jsonable_dict['classifiers_optimized'] =\
            {i: classifier_to_jsonable_dict(clf, skipkeys=common_meta_keys)
             for i, clf in self.MLdict.items()}

        # add common metadata and corresponding decoding type hints
        for k in common_meta_keys:
            jsonable_dict[k], decode_type = get_jsonable_value(getattr(self, k), return_typesdict=True)

            if decode_type:
                decode_types_dict[k] = decode_type

        jsonable_dict['__decode_types'] = decode_types_dict

        return jsonable_dict

    # def save_to_json(self, filepath):
    #     dict2save = dict(
    #         cluster_centers=self.cluster_centers.tolist(),
    #
    #     )
    #
    #     # Create json and save to file
    #     json_txt = json.dumps(dict2save, indent=4)
    #     with open(filepath, 'w') as file:
    #         file.write(json_txt)


class ClassifierCollection(object):
    def __init__(self, path_dillFile):
        with open(path_dillFile, 'rb') as inF:
            self.content = dill.load(inF)

    def __repr__(self):
        """Return the representation of ClassifierCollection.

        :return: e.g., "{'1__2__3__4__5__7': {('Landsat-5', 'TM'): {'1__2__3__4__5__7':
                        LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)}, ..."
        """
        return pformat(self.content)

    def __getitem__(self, item):
        """Get a specific item of the ClassifierCollection."""
        try:
            return self.content[item]
        except KeyError:
            raise(KeyError("The classifier has no key '%s'. Available keys are: %s"
                           % (item, repr(self))))

    # def save_to_json(self, filepath):
    #     a = 1
    #     pass


def get_jsonable_value(in_value, return_typesdict=False):
    if isinstance(in_value, np.ndarray):
        outval = in_value.tolist()
    elif isinstance(in_value, list):
        outval = np.array(in_value).tolist()
        # json.dumps(outval)
    else:
        outval = in_value

    # FIXME: In case of quadratic regression, there are some attributes that are not directly JSONable in this manner.

    # create a dictionary containing the data types needed for JSON decoding
    typesdict = dict()
    if return_typesdict and not isinstance(in_value, (str, int, float, bool)) and in_value is not None:
        typesdict['type'] = type(in_value).__name__

        if isinstance(in_value, np.ndarray):
            typesdict['dtype'] = in_value.dtype.name

        if isinstance(in_value, list):
            typesdict['dtype'] = type(in_value[0]).__name__

            if not len(set(type(vv).__name__ for vv in in_value)) == 1:
                raise RuntimeError('Lists containing different data types of list elements cannot be made '
                                   'jsonable without losses.')

    if return_typesdict:
        return outval, typesdict
    else:
        return outval


def classifier_to_jsonable_dict(clf, skipkeys: list = None, include_typesdict=False):
    from sklearn.linear_model import LinearRegression  # avoids static TLS error here

    if isinstance(clf, LinearRegression):
        jsonable_dict = dict(clftype='LR')
        typesdict = dict()

        for k, v in clf.__dict__.items():
            if skipkeys and k in skipkeys:
                continue

            if include_typesdict:
                jsonable_dict[k], typesdict[k] = get_jsonable_value(v, return_typesdict=True)
            else:
                jsonable_dict[k] = get_jsonable_value(v)

            # if valtype is np.ndarray:
            #     jsonable_dict[k] = dict(val=v.tolist(),
            #                        dtype=v.dtype.name)
            # elif valtype is list:
            #     jsonable_dict[k] = dict(val=np.array(v).tolist())
            # else:
            #     jsonable_dict[k] = dict(val=v)
            #
            # jsonable_dict[k]['valtype'] = valtype.__name__

    else:  # Ridge, Pipeline, RandomForestRegressor:
        # TODO
        raise NotImplementedError('At the moment, only LR classifiers can be serialized to JSON format.')

    if include_typesdict:
        jsonable_dict['__decode_types'] = {k: v for k, v in typesdict.items() if v}

    return jsonable_dict


def classifier_from_json_str(json_str):
    """Create a spectral harmonization classifier from a JSON string (JSON de-serialization).

    :param json_str:    the JSON string to be used for de-serialization
    :return:
    """
    from sklearn.linear_model import LinearRegression  # avoids static TLS error here

    in_dict = json.loads(json_str)

    if in_dict['clftype']['val'] == 'LR':
        clf = LinearRegression()
    else:
        raise NotImplementedError("Unknown object type '%s'." % in_dict['objecttype'])

    for k, v in in_dict.items():
        try:
            val2set = getattr(builtins, v['valtype'])(v['val'])
        except (AttributeError, KeyError):
            if v['valtype'] == 'ndarray':
                val2set = np.array(v['val']).astype(np.dtype(v['dtype']))
            else:
                raise TypeError("Unexpected object type '%s'." % v['valtype'])

        setattr(clf, k, val2set)

    return clf
