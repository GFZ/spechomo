# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from typing import Union  # noqa F401  # flake8 issue

import numpy as np
from geoarray import GeoArray
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans


class KMeansRSImage(object):
    """Class for clustering a given input image by using K-Means algorithm.

    NOTE: Based on the nodata value of the input GeoArray those pixels that have nodata values in some bands are
          ignored when computing the cluster coefficients. Nodata values would affect clustering result otherwise.
    """
    def __init__(self, im, n_clusters, CPUs=1, v=False):
        # type: (GeoArray, int, Union[None, int], bool) -> None

        # privates
        self._clusters = None
        self._im_clust = None
        self._spectra = None

        self.im = im
        self.n_clusters = n_clusters
        self.CPUs = CPUs or cpu_count()
        self.goodSpecMask = None
        self.v = v

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
    def im_clust(self):
        if self._im_clust is None:
            self._im_clust = self.get_labels_with_nodata().reshape((self.im.rows, self.im.cols))

        return self._im_clust

    def compute_clusters(self):
        spectra = self._im2spectra(self.im)

        # filter spectra containing no data values
        # to prevent pixels containing no data values from beeing included in the clustering
        if self.im.nodata is not None:
            mask_nodata = spectra == self.im.nodata
            self.goodSpecMask = np.all(~mask_nodata, axis=1)

            if True not in self.goodSpecMask:
                raise RuntimeError('All spectra contain no data values in one or multiple bands and are therefore not '
                                   'usable for clustering. Clustering failed.')

            spectra = spectra[self.goodSpecMask, :]

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_jobs=self.CPUs, verbose=self.v)
        self.clusters = kmeans.fit(spectra)

        return self.clusters

    def get_labels_with_nodata(self):
        """Get the labels with respect to nodata values."""
        if self.goodSpecMask is None:
            return self.clusters.labels_
        else:
            labels = np.full_like(self.goodSpecMask, -9999, dtype=self.clusters.labels_.dtype)
            labels[self.goodSpecMask] = self.clusters.labels_
            return labels

    def apply_clusters(self, image):
        labels = self.clusters.predict(self._im2spectra(GeoArray(image)))
        return labels

    @staticmethod
    def _im2spectra(geoArr):
        return geoArr.reshape((geoArr.rows * geoArr.cols, geoArr.bands))

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

    def plot_clustered_image(self, figsize=None):
        # type: (tuple) -> None
        """Show a the clustered image.

        :param figsize:     figure size (inches)
        """
        plt.figure(figsize=figsize)
        rows, cols = self.im_clust.shape[:2]

        image2plot = self.im_clust if self.im.nodata is None else np.ma.masked_equal(self.im_clust, self.im.nodata)

        plt.imshow(image2plot, plt.get_cmap('prism'), interpolation='none', extent=(0, cols, rows, 0))
        plt.show()

    def get_random_spectra_from_each_cluster(self, samplesize=50, src_im=None, exclude_worst_percent=None):
        # type: (int, GeoArray, int) -> dict
        """Returns a given number of spectra randomly selected within each cluster.

        E.g., 50 spectra belonging to cluster 1, 50 spectra belonging to cluster 2 and so on.

        :param samplesize:  number of spectra to be randomly selected from each cluster
        :param src_im:      image to get random samples from (default: self.im)
        :param exclude_worst_percent:   percentage of spectra with the largest spectral distances to be excluded
                                        from random sampling
        :return:
        """
        src_im = src_im if src_im is not None else self.im

        if not self._clusters:
            self.compute_clusters()

        # get DataFrame with columns [cluster_label, B1, B2, B3, ...]
        if self.goodSpecMask is None:
            src_spectra = self._im2spectra(src_im)
        else:
            src_spectra = self._im2spectra(src_im)[self.goodSpecMask, :]
        df = DataFrame(src_spectra, columns=['B%s' % band for band in range(1, src_im.bands + 1)], )
        df.insert(0, 'cluster_label', self.clusters.labels_)

        if exclude_worst_percent is not None:
            if not 0 < exclude_worst_percent < 100:
                raise ValueError(exclude_worst_percent)
            df.insert(1, 'spectral_distance', np.min(self.clusters.fit_transform(src_spectra), axis=1))

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

    def get_purest_spectra_from_each_cluster(self, samplesize=50, src_im=None):
        # type: (int, GeoArray) -> dict
        """Returns a given number of spectra directly surrounding the center of each cluster.

        E.g., 50 spectra belonging to cluster 1, 50 spectra belonging to cluster 2 and so on.

        :param samplesize:  number of spectra to be selected from each cluster
        :param src_im:      image to get samples from (default: self.im)
        :return:
        """
        src_im = src_im if src_im is not None else self.im

        if not self._clusters:
            self.compute_clusters()

        # get DataFrame with columns [cluster_label, B1, B2, B3, ...]
        if self.goodSpecMask is None:
            src_spectra = self._im2spectra(src_im)
        else:
            src_spectra = self._im2spectra(src_im)[self.goodSpecMask, :]
        df = DataFrame(src_spectra, columns=['B%s' % band for band in range(1, src_im.bands + 1)], )
        df.insert(0, 'cluster_label', self.clusters.labels_)
        df.insert(1, 'spectral_distance', np.min(self.clusters.fit_transform(src_spectra), axis=1))

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
