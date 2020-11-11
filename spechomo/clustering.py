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

from multiprocessing import cpu_count
from typing import Union, TYPE_CHECKING  # noqa F401  # flake8 issue
import os

if TYPE_CHECKING:
    from sklearn.cluster import KMeans

import dill
import numpy as np
from geoarray import GeoArray
from pandas import DataFrame
from specclassify import SAM_Classifier, classify_image

from .utils import im2spectra


class KMeansRSImage(object):
    """Class for clustering a given input image by using K-Means algorithm.

    NOTE: Based on the nodata value of the input GeoArray those pixels that have nodata values in some bands are
          ignored when computing the cluster coefficients. Nodata values would affect clustering result otherwise.
    """

    def __init__(self, im, n_clusters, sam_classassignment=False, CPUs=1, v=False):
        # type: (GeoArray, int, bool, Union[None, int], bool) -> None

        # privates
        self._clusters = None
        self._clustermap = None
        self._goodSpecMask = None
        self._spectra = None
        self._labels_with_nodata = None
        self._spectral_distances = None
        self._spectral_distances_with_nodata = None
        self._spectral_angles = None
        self._spectral_angles_with_nodata = None

        self.im = im
        self.n_clusters = n_clusters
        self.sam_classassignment = sam_classassignment
        self.CPUs = CPUs or cpu_count()
        self.v = v

    @classmethod
    def from_disk(cls, path_clf, im):
        # type: (str, GeoArray) -> KMeansRSImage
        """Get an instance of KMeansRSImage from a previously saved classifier.

        :param path_clf:    path of serialzed classifier (dill file)
        :param im:          path of the image cube belonging to that classifier
        :return: KMeansRSImage
        """
        with open(path_clf, 'rb') as inF:
            undilled = dill.load(inF)  # type: dict

        KM = KMeansRSImage(im,
                           undilled['clusters'].n_clusters,
                           undilled['clusters'].n_jobs,
                           undilled['clusters'].verbose)
        KM._clusters = undilled['clusters']
        KM._goodSpecMask = undilled['_goodSpecMask']
        KM._spectral_distances = undilled['_spectral_distances']

        return KM

    @property
    def goodSpecMask(self):
        if self._goodSpecMask is None:
            if self.im.nodata is not None:
                mask_nodata = im2spectra(self.im) == self.im.nodata
                goodSpecMask = np.all(~mask_nodata, axis=1)

                if True not in goodSpecMask:
                    raise RuntimeError('All spectra contain no data values in one or multiple bands and are therefore '
                                       'not usable for clustering. Clustering failed.')

                self._goodSpecMask = goodSpecMask
            else:
                self._goodSpecMask = np.ones((self.im.rows * self.im.cols), dtype=np.bool)

        return self._goodSpecMask

    @property
    def spectra(self):
        """Get spectra used for clustering (excluding spectra containing nodata values that would affect clustering)."""
        if self._spectra is None:
            self._spectra = im2spectra(self.im)[self.goodSpecMask, :]
        return self._spectra

    @property
    def n_spectra(self):
        """Get number of spectra used for clustering (excluding spectra containing nodata values)."""
        return int(np.sum(self.goodSpecMask))

    @property
    def clusters(self):
        # type: () -> KMeans
        if not self._clusters:
            self._clusters = self.compute_clusters()
        return self._clusters

    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters

    @property
    def clustermap(self):
        if self._clustermap is None:
            self._clustermap = self.labels_with_nodata.reshape((self.im.rows, self.im.cols))

        return self._clustermap

    def compute_clusters(self, nmax_spectra=100000):
        """Compute the cluster means and labels.

        :param nmax_spectra:    maximum number of spectra to be included (pseudo-randomly selected (reproducable))
        :return:
        """
        from sklearn.cluster import KMeans  # avoids static TLS ImportError here
        from sklearn import __version__ as skver

        _old_environ = dict(os.environ)

        try:
            kwargs_kmeans = dict(n_clusters=self.n_clusters, random_state=0, verbose=self.v)
            # scikit-learn>0.23 uses all cores by default; number is adjustable via OMP_NUM_THREADS
            if float('%s.%s' % tuple(skver.split('.')[:2])) < 0.23:
                kwargs_kmeans['n_jobs'] = self.CPUs
            else:
                os.environ['OMP_NUM_THREADS '] = str(self.CPUs)

            # data reduction in case we have too many spectra
            if self.spectra.shape[0] > nmax_spectra:
                if self.v:
                    print('Reducing data...')
                idxs_specIncl = np.random.RandomState(seed=0).choice(range(self.n_spectra), nmax_spectra)
                idxs_specNotIncl = np.array(range(self.n_spectra))[~np.in1d(range(self.n_spectra), idxs_specIncl)]
                spectra_incl = self.spectra[idxs_specIncl, :]
                spectra_notIncl = self.spectra[idxs_specNotIncl, :]

                if self.v:
                    print('Fitting KMeans...')
                kmeans = KMeans(**kwargs_kmeans)
                distmatrix_incl = kmeans.fit_transform(spectra_incl)

                if self.v:
                    print('Computing full resolution labels...')
                labels = np.zeros((self.n_spectra,), dtype=kmeans.labels_.dtype)
                distances = np.zeros((self.n_spectra,), dtype=distmatrix_incl.dtype)

                labels[idxs_specIncl] = kmeans.labels_
                distances[idxs_specIncl] = np.min(distmatrix_incl, axis=1)

                if self.sam_classassignment:
                    # override cluster labels with labels computed via SAM (distances have be recomputed then)
                    print('Using SAM class assignment.')
                    SC = SAM_Classifier(kmeans.cluster_centers_, CPUs=self.CPUs)
                    im_sam_labels = SC.classify(self.im)
                    sam_labels = im_sam_labels.flatten()[self.goodSpecMask]
                    self._spectral_angles = SC.angles_deg.flatten()[self.goodSpecMask]

                    # update distances at those positions where SAM assigns different class labels
                    distsPos2update = labels != sam_labels
                    distsPos2update[idxs_specNotIncl] = True
                    distances[distsPos2update] = \
                        self.compute_euclidian_distance_for_labelled_spectra(
                            self.spectra[distsPos2update, :], sam_labels[distsPos2update], kmeans.cluster_centers_)

                    kmeans.labels_ = sam_labels

                else:
                    distmatrix_specNotIncl = kmeans.transform(spectra_notIncl)
                    labels[idxs_specNotIncl] = np.argmin(distmatrix_specNotIncl, axis=1)
                    distances[idxs_specNotIncl] = np.min(distmatrix_specNotIncl, axis=1)

                    kmeans.labels_ = labels

            else:
                if self.v:
                    print('Fitting KMeans...')
                kmeans = KMeans(**kwargs_kmeans)

                distmatrix = kmeans.fit_transform(self.spectra)
                kmeans_labels = kmeans.labels_
                distances = np.min(distmatrix, axis=1)

                if self.sam_classassignment:
                    # override cluster labels with labels computed via SAM (distances have be recomputed then)
                    print('Using SAM class assignment.')
                    SC = SAM_Classifier(kmeans.cluster_centers_, CPUs=self.CPUs)
                    im_sam_labels = SC.classify(self.im)
                    sam_labels = im_sam_labels.flatten()[self.goodSpecMask]
                    self._spectral_angles = SC.angles_deg.flatten()[self.goodSpecMask]

                    # update distances at those positions where SAM assigns different class labels
                    distsPos2update = kmeans_labels != sam_labels
                    distances[distsPos2update] = \
                        self.compute_euclidian_distance_for_labelled_spectra(
                            self.spectra[distsPos2update, :], sam_labels[distsPos2update], kmeans.cluster_centers_)
        finally:
            os.environ.clear()
            os.environ.update(_old_environ)

        self.clusters = kmeans
        self._spectral_distances = distances

        return self.clusters

    def apply_clusters(self, image):
        labels = self.clusters.predict(im2spectra(GeoArray(image)))
        return labels

    def compute_spectral_distances(self):
        self._spectral_distances = np.min(self.clusters.transform(self.spectra), axis=1)
        return self.spectral_distances

    @staticmethod
    def compute_euclidian_distance_2D(spectra, endmembers):
        n_samples, n_features = endmembers.shape

        if not spectra.shape[1] == endmembers.shape[1]:
            raise RuntimeError('Matrix dimensions are not aligned. Input spectra have %d bands but endmembers '
                               'have %d.' % (spectra.shape[1], endmembers.shape[1]))

        dists = np.zeros((spectra.shape[0], n_samples), np.float32)

        # loop over all endmember spectra and compute spectral angle for each input spectrum
        for n_sample in range(n_samples):
            train_spectrum = endmembers[n_sample, :].astype(np.float)
            diff = spectra - train_spectrum
            dists[:, n_sample] = np.sqrt((diff ** 2).sum(axis=1))

        return dists

    @staticmethod
    def compute_euclidian_distance_for_labelled_spectra(spectra, labels, endmembers):
        if not spectra.shape[1] == endmembers.shape[1]:
            raise RuntimeError('Matrix dimensions are not aligned. Input spectra have %d bands but endmembers '
                               'have %d.' % (spectra.shape[1], endmembers.shape[1]))

        dists = np.zeros((spectra.shape[0]), np.float32)

        # loop over all endmember spectra and compute spectral angle for each input spectrum
        for lbl in np.unique(labels):
            train_spectrum = endmembers[lbl, :].astype(np.float)
            mask_curlbl = labels == lbl
            spectra_curlbl = spectra[mask_curlbl, :]
            diff_curlbl = spectra_curlbl - train_spectrum
            dists[mask_curlbl] = np.sqrt((diff_curlbl ** 2).sum(axis=1))

        return dists

    def compute_spectral_angles(self):
        spectral_angles = classify_image(self.im, self.clusters.cluster_centers_, list(range(self.n_clusters)),
                                         'SAM', in_nodataVal=self.im.nodata, cmap_nodataVal=-9999,
                                         tiledims=(400, 400), CPUs=self.CPUs, return_distance=True,
                                         unclassified_pixVal=-1)[1]

        self._spectral_angles = spectral_angles.flatten()[self.goodSpecMask]
        return self.spectral_angles

    @property
    def labels(self):
        """Get labels for all clustered spectra (excluding spectra that contain nodata values)."""
        return self.clusters.labels_

    @property
    def labels_with_nodata(self):
        """Get the labels for all pixels (including those containing nodata values)."""
        if self._labels_with_nodata is None:
            if self.n_spectra == (self.im.rows * self.im.cols):
                self._labels_with_nodata = self.clusters.labels_
            else:
                labels = np.full_like(self.goodSpecMask, -9999, dtype=self.clusters.labels_.dtype)
                labels[self.goodSpecMask] = self.clusters.labels_
                self._labels_with_nodata = labels

        return self._labels_with_nodata

    @property
    def spectral_distances(self):
        """Get spectral distances for all pixels that don't contain nodata values."""
        if self._spectral_distances is None:
            self._spectral_distances = self.compute_spectral_distances()

        return self._spectral_distances

    @property
    def spectral_distances_with_nodata(self):
        if self._clusters is not None and self._spectral_distances_with_nodata is None:
            if self.n_spectra == (self.im.rows * self.im.cols):
                self._spectral_distances_with_nodata = self.spectral_distances
            else:
                dists = np.full_like(self.goodSpecMask, np.nan, dtype=np.float)
                dists[self.goodSpecMask] = self.spectral_distances
                self._spectral_distances_with_nodata = dists

        return self._spectral_distances_with_nodata

    @property
    def spectral_angles(self):
        """Get spectral angles in degrees for all pixels that don't contain nodata values."""
        if self._spectral_angles is None:
            self._spectral_angles = self.compute_spectral_angles()

        return self._spectral_angles

    @property
    def spectral_angles_with_nodata(self):
        if self._clusters is not None and self._spectral_angles_with_nodata is None:
            if self.n_spectra == (self.im.rows * self.im.cols):
                self._spectral_angles_with_nodata = self.spectral_angles
            else:
                angles = np.full_like(self.goodSpecMask, np.nan, dtype=np.float)
                angles[self.goodSpecMask] = self.spectral_angles
                self._spectral_angles_with_nodata = angles

        return self._spectral_angles_with_nodata

    def dump(self, path_out):
        with open(path_out, 'wb') as outF:
            dill.dump(dict(
                clusters=self.clusters,
                _goodSpecMask=self._goodSpecMask,
                _spectral_distances=self._spectral_distances),
                outF)

    def save_clustermap(self, path_out, **kw_save):
        GeoArray(self.clustermap, geotransform=self.im.gt, projection=self.im.prj, nodata=-9999)\
            .save(path_out, **kw_save)

    def plot_cluster_centers(self, figsize=(15, 5)):
        # type: (tuple) -> None
        """Show a plot of the cluster center signatures.

        :param figsize:     figure size (inches)
        """
        from matplotlib import pyplot as plt

        plt.figure(figsize=figsize)
        for i, center_signature in enumerate(self.clusters.cluster_centers_):
            plt.plot(range(1, self.im.bands + 1), center_signature, label='Cluster #%s' % (i + 1))

        plt.title('KMeans cluster centers for %s clusters' % self.n_clusters)
        plt.xlabel('Spectral band')
        plt.ylabel('Pixel values')
        plt.legend(loc='upper right')

        plt.show()

    def plot_cluster_histogram(self, figsize=(15, 5)):
        # type: (tuple) -> None
        """Show a histogram indicating the proportion of each cluster label in percent.

        :param figsize:     figure size (inches)
        """
        from matplotlib import pyplot as plt

        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(self.clusters.labels_)) + 1)
        hist, bins = np.histogram(self.clusters.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to 100
        hist = hist.astype("float")
        hist /= hist.sum() / 100

        # plot the histogram as bar plot
        plt.figure(figsize=figsize)

        plt.bar(bins[:-1], hist, width=1)

        plt.title('Proportion of cluster labels (%s clusters)' % self.n_clusters)
        plt.xlabel('# Cluster')
        plt.ylabel('Proportion [%]')

        plt.show()

    def plot_clustermap(self, figsize=None):
        # type: (tuple) -> None
        """Show a the clustered image.

        :param figsize:     figure size (inches)
        """
        from matplotlib import pyplot as plt

        plt.figure(figsize=figsize)
        rows, cols = self.clustermap.shape[:2]

        image2plot = self.clustermap if self.im.nodata is None else np.ma.masked_equal(self.clustermap, self.im.nodata)

        plt.imshow(image2plot, plt.get_cmap('prism'), interpolation='none', extent=(0, cols, rows, 0))
        plt.show()

    def get_random_spectra_from_each_cluster(self, samplesize=50, max_distance=None, max_angle=None,
                                             nmin_unique_spectra=50):
        # type: (int, Union[int, float, str], Union[int, float, str], int) -> dict
        """Return a given number of spectra randomly selected within each cluster.

        E.g., 50 spectra belonging to cluster 1, 50 spectra belonging to cluster 2 and so on.

        :param samplesize:  number of spectra to be randomly selected from each cluster
        :param max_distance:    spectra with a larger spectral distance than the given value will be excluded from
                                random sampling.
                                - if given as string like '20%', the maximum spectral distance is computed as 20%
                                percentile within each cluster
        :param max_angle:       spectra with a larger spectral angle than the given value will be excluded from
                                random sampling.
                                - if given as string like '20%', the maximum spectral angle is computed as 20%
                                percentile within each cluster
        :param nmin_unique_spectra:   in case a cluster has less than the given number, do not use its spectra
                                      (return missing values)
        :return:
        """
        # get DataFrame with columns [cluster_label, B1, B2, B3, ...]
        df = DataFrame(self.spectra, columns=['B%s' % band for band in range(1, self.im.bands + 1)], )
        df.insert(0, 'cluster_label', self.clusters.labels_)

        if max_angle is not None:
            if not (isinstance(max_angle, (int, float)) and max_angle > 0) and \
               not (isinstance(max_angle, str) and max_angle.endswith('%')):
                raise ValueError(max_angle)
            if isinstance(max_angle, str):
                max_angle = np.percentile(self.spectral_angles, float(max_angle.split('%')[0].strip()))
            df.insert(1, 'spectral_angle', self.spectral_angles)

        if max_distance is not None:
            if not (isinstance(max_distance, (int, float)) and 0 < max_distance < 100) and \
               not (isinstance(max_distance, str) and max_distance.endswith('%')):
                raise ValueError(max_distance)
            if isinstance(max_distance, str):
                max_distance = np.percentile(self.spectral_distances, float(max_distance.split('%')[0].strip()))
            df.insert(1, 'spectral_distance', self.spectral_distances)

        # get random sample from each cluster and generate a dict like {cluster_label: random_sample}
        # NOTE: nodata label is skipped
        random_samples = dict()
        for label in range(self.n_clusters):
            if max_angle is None and max_distance is None:
                cluster_subset = df[df.cluster_label == label].loc[:, 'B1':]
            else:
                cluster_subset = df[df.cluster_label == label]

                # if self.n_clusters > 1:
                # filter by spectral angle
                if len(cluster_subset.index) >= nmin_unique_spectra and max_angle is not None:
                    cluster_subset = cluster_subset[cluster_subset.spectral_angle < max_angle]

                # filter by spectral distance
                if len(cluster_subset.index) >= nmin_unique_spectra and max_distance is not None:
                    cluster_subset = cluster_subset[cluster_subset.spectral_distance < max_distance]

                cluster_subset = cluster_subset.loc[:, 'B1':]

                # handle clusters with less than nmin_unique_spectra in there
                if len(cluster_subset.index) < nmin_unique_spectra:
                    if len(cluster_subset.index) > 0:
                        # don't use the cluster (return nodata)
                        cluster_subset[:] = -9999

                    else:
                        # cluster_subset is empty after filtering -> return nodata
                        cluster_subset.loc[0] = [-9999] * len(cluster_subset.columns)

            # get random sample while filling it with duplicates of the same sample when cluster has not enough spectra
            random_samples[label] = np.array(cluster_subset.sample(samplesize, replace=True, random_state=20))

        return random_samples

    def get_purest_spectra_from_each_cluster(self, samplesize=50):
        # type: (int) -> dict
        """Return a given number of spectra directly surrounding the center of each cluster.

        E.g., 50 spectra belonging to cluster 1, 50 spectra belonging to cluster 2 and so on.

        :param samplesize:  number of spectra to be selected from each cluster
        :return:
        """
        # get DataFrame with columns [cluster_label, B1, B2, B3, ...]
        df = DataFrame(self.spectra, columns=['B%s' % band for band in range(1, self.im.bands + 1)], )
        df.insert(0, 'cluster_label', self.clusters.labels_)
        df.insert(1, 'spectral_distance', self.spectral_distances)

        # get random sample from each cluster and generate a dict like {cluster_label: random_sample}
        random_samples = dict()
        for label in range(self.n_clusters):
            cluster_subset = df[df.cluster_label == label].loc[:, 'spectral_distance':]

            # catch the case that the cluster does not contain enough spectra (duplicate existing ones)
            if len(cluster_subset.index) < samplesize:
                cluster_subset = cluster_subset.sample(samplesize, replace=True, random_state=20)

            cluster_subset_sorted = cluster_subset.sort_values(by=['spectral_distance'], ascending=True)
            random_samples[label] = np.array(cluster_subset_sorted.loc[:, 'B1':][:samplesize])

        return random_samples
