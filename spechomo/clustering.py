# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from typing import Union  # noqa F401  # flake8 issue

import dill
import numpy as np
from geoarray import GeoArray
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans

from .utils import im2spectra


class KMeansRSImage(object):
    """Class for clustering a given input image by using K-Means algorithm.

    NOTE: Based on the nodata value of the input GeoArray those pixels that have nodata values in some bands are
          ignored when computing the cluster coefficients. Nodata values would affect clustering result otherwise.
    """
    def __init__(self, im, n_clusters, CPUs=1, v=False):
        # type: (GeoArray, int, Union[None, int], bool) -> None

        # privates
        self._clusters = None
        self._clustermap = None
        self._goodSpecMask = None
        self._spectra = None
        self._labels_with_nodata = None
        self._spectral_distances = None
        self._spectral_distances_with_nodata = None

        self.im = im
        self.n_clusters = n_clusters
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

    def compute_clusters(self, nmax_spectra=1000000):
        """Compute the cluster means and labels.

        :param nmax_spectra:    maximum number of spectra to be included (pseudo-randomly selected (reproducable))
        :return:
        """
        # data reduction in case we have too many spectra
        if self.spectra.shape[0] > 1e6:
            if self.v:
                print('Reducing data...')
            idxs_specIncl = np.random.RandomState(seed=0).choice(range(self.n_spectra), nmax_spectra)
            idxs_specNotIncl = np.array(range(self.n_spectra))[~np.in1d(range(self.n_spectra), idxs_specIncl)]
            spectra_incl = self.spectra[idxs_specIncl, :]
            spectra_notIncl = self.spectra[idxs_specNotIncl, :]

            if self.v:
                print('Fitting KMeans...')
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_jobs=self.CPUs, verbose=self.v)
            kmeans.fit(spectra_incl)

            if self.v:
                print('Computing full resolution labels...')
            labels = np.zeros((self.n_spectra,), dtype=kmeans.labels_.dtype)
            labels[idxs_specIncl] = kmeans.labels_
            labels[idxs_specNotIncl] = kmeans.predict(spectra_notIncl)

            kmeans.labels_ = labels

        else:
            if self.v:
                print('Fitting KMeans...')
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_jobs=self.CPUs, verbose=self.v)
            kmeans.fit(self.spectra)

        self.clusters = kmeans

        return self.clusters

    def apply_clusters(self, image):
        labels = self.clusters.predict(im2spectra(GeoArray(image)))
        return labels

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
        # TODO compute that in multiprocessing
        if self._spectral_distances is None:
            self._spectral_distances = np.min(self.clusters.fit_transform(self.spectra), axis=1)

        return self._spectral_distances

    @property
    def spectral_distances_with_nodata(self):
        if self._spectral_distances_with_nodata is None:
            if self.n_spectra == (self.im.rows * self.im.cols):
                self._spectral_distances_with_nodata = self.spectral_distances
            else:
                dists = np.full_like(self.goodSpecMask, np.nan, dtype=np.float)
                dists[self.goodSpecMask] = self.spectral_distances
                self._spectral_distances_with_nodata = dists

        return self._spectral_distances_with_nodata

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
        plt.figure(figsize=figsize)
        rows, cols = self.clustermap.shape[:2]

        image2plot = self.clustermap if self.im.nodata is None else np.ma.masked_equal(self.clustermap, self.im.nodata)

        plt.imshow(image2plot, plt.get_cmap('prism'), interpolation='none', extent=(0, cols, rows, 0))
        plt.show()

    def get_random_spectra_from_each_cluster(self, samplesize=50, exclude_worst_percent=None):
        # type: (int, int) -> dict
        """Returns a given number of spectra randomly selected within each cluster.

        E.g., 50 spectra belonging to cluster 1, 50 spectra belonging to cluster 2 and so on.

        :param samplesize:  number of spectra to be randomly selected from each cluster
        :param exclude_worst_percent:   percentage of spectra with the largest spectral distances to be excluded
                                        from random sampling
        :return:
        """
        # get DataFrame with columns [cluster_label, B1, B2, B3, ...]
        df = DataFrame(self.spectra, columns=['B%s' % band for band in range(1, self.im.bands + 1)], )
        df.insert(0, 'cluster_label', self.clusters.labels_)

        if exclude_worst_percent is not None:
            if not 0 < exclude_worst_percent < 100:
                raise ValueError(exclude_worst_percent)
            df.insert(1, 'spectral_distance', self.spectral_distances)

        # get random sample from each cluster and generate a dict like {cluster_label: random_sample}
        # NOTE: nodata label is skipped
        random_samples = dict()
        for label in range(self.n_clusters):
            if exclude_worst_percent is None:
                cluster_subset = df[df.cluster_label == label].loc[:, 'B1':]
            else:
                cluster_subset = df[df.cluster_label == label].loc[:, 'spectral_distance':]
                cluster_subset.sort_values(by=['spectral_distance'], ascending=True)
                min_th = np.percentile(cluster_subset.spectral_distance, exclude_worst_percent)
                cluster_subset = cluster_subset[cluster_subset.spectral_distance < min_th].loc[:, 'B1':]

            # get random sample while filling it with duplicates of the same sample when cluster has not enough spectra
            random_samples[label] = np.array(cluster_subset.sample(samplesize, replace=True, random_state=20))

        return random_samples

    def get_purest_spectra_from_each_cluster(self, samplesize=50):
        # type: (int) -> dict
        """Returns a given number of spectra directly surrounding the center of each cluster.

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
