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

import json
import os
import re
from collections import OrderedDict
from typing import Union, List  # noqa F401  # flake8 issue
from tqdm import tqdm

import numpy as np
from geoarray import GeoArray
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from pyrsr import RSR

from .utils import im2spectra


class TrainingData(object):
    """Class for analyzing statistical relations between a pair of machine learning training data cubes."""

    def __init__(self, im_X, im_Y, test_size):
        # type: (Union[GeoArray, np.ndarray], Union[GeoArray, np.ndarray], Union[float, int]) -> None
        """Get instance of TrainingData.

        :param im_X:        input image X
        :param im_Y:        input image Y
        :param test_size:   test size (proportion as float between 0 and 1) or absolute value as integer
        """
        from sklearn.model_selection import train_test_split  # avoids static TLS error here

        self.im_X = GeoArray(im_X)
        self.im_Y = GeoArray(im_Y)

        # Set spectra (3D to 2D conversion)
        self.spectra_X = im2spectra(self.im_X)
        self.spectra_Y = im2spectra(self.im_Y)

        # Set train and test variables
        # NOTE: If random_state is set to an Integer, train_test_split will always select the same 'pseudo-random' set
        #       of the input data.
        self.train_X, self.test_X, self.train_Y, self.test_Y = \
            train_test_split(self.spectra_X, self.spectra_Y, test_size=test_size, shuffle=True, random_state=0)

    def plot_scatter_matrix(self, figsize=(15, 15), mode='intersensor'):
        # TODO complete this function
        from matplotlib import pyplot as plt

        train_X = self.train_X[np.random.choice(self.train_X.shape[0], 1000, replace=False), :]
        train_Y = self.train_Y[np.random.choice(self.train_Y.shape[0], 1000, replace=False), :]

        if mode == 'intersensor':
            import seaborn

            fig, axes = plt.subplots(train_X.shape[1], train_Y.shape[1],
                                     figsize=(25, 9), sharex='all', sharey='all')
            # fig.suptitle('Correlation of %s and %s bands' % (self.src_cube.satellite, self.tgt_cube.satellite),
            #              size=25)

            color = seaborn.hls_palette(13)

            for i, ax in zip(range(train_X.shape[1]), axes.flatten()):
                for j, ax in zip(range(train_Y.shape[1]), axes.flatten()):
                    axes[i, j].scatter(train_X[:, j], train_Y[:, i], c=color[j], label=str(j))
                    # axes[i, j].set_xlim(-0.1, 1.1)
                    # axes[i, j].set_ylim(-0.1, 1.1)
                    #  if j == 8:
                    #      axes[5, j].set_xlabel('S2 B8A\n' + str(metadata_s2['Bands_S2'][j]) + ' nm', size=10)
                    #  elif j in range(9, 13):
                    #      axes[5, j].set_xlabel('S2 B' + str(j) + '\n' + str(metadata_s2['Bands_S2'][j]) + ' nm',
                    #                            size=10)
                    #  else:
                    #      axes[5, j].set_xlabel('S2 B' + str(j + 1) + '\n' + str(metadata_s2['Bands_S2'][j]) + ' nm',
                    #                            size=10)
                    #  axes[i, 0].set_ylabel(
                    #      'S3 SLSTR B' + str(6 - i) + '\n' + str(metadata_s3['Bands_S3'][5 - i]) + ' nm',
                    #      size=10)
                    # axes[4, j].set_xticks(np.arange(0, 1.2, 0.2))
                    # axes[i, j].plot([0, 1], [0, 1], c='red')

        else:
            df = DataFrame(train_X, columns=['Band %s' % b for b in range(1, self.im_X.bands + 1)])
            scatter_matrix(df, figsize=figsize, marker='.', hist_kwds={'bins': 50}, s=30, alpha=0.8)
            plt.suptitle('Image X band to band correlation')

            df = DataFrame(train_Y, columns=['Band %s' % b for b in range(1, self.im_Y.bands + 1)])
            scatter_matrix(df, figsize=figsize, marker='.', hist_kwds={'bins': 50}, s=30, alpha=0.8)
            plt.suptitle('Image Y band to band correlation')

    def plot_scattermatrix(self):
        # TODO complete this function
        import seaborn
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(self.im_X.data.bands, self.im_Y.data.bands,
                                 figsize=(25, 9), sharex='all', sharey='all')
        fig.suptitle('Correlation of %s and %s bands' % (self.im_X.satellite, self.im_Y.satellite), size=25)

        color = seaborn.hls_palette(13)

        for i, ax in zip(range(6), axes.flatten()):
            for j, ax in zip(range(13), axes.flatten()):
                axes[i, j].scatter(self.train_X[:, j], self.train_Y[:, 5 - i], c=color[j], label=str(j))
                axes[i, j].set_xlim(-0.1, 1.1)
                axes[i, j].set_ylim(-0.1, 1.1)
                # if j == 8:
                #     axes[5, j].set_xlabel('S2 B8A\n' + str(metadata_s2['Bands_S2'][j]) + ' nm', size=10)
                # elif j in range(9, 13):
                #     axes[5, j].set_xlabel('S2 B' + str(j) + '\n' + str(metadata_s2['Bands_S2'][j]) + ' nm', size=10)
                # else:
                #     axes[5, j].set_xlabel('S2 B' + str(j + 1) + '\n' + str(metadata_s2['Bands_S2'][j]) + ' nm',
                #                           size=10)
                # axes[i, 0].set_ylabel('S3 SLSTR B' + str(6 - i) + '\n' + str(metadata_s3['Bands_S3'][5 - i]) + ' nm',
                #                       size=10)
                axes[4, j].set_xticks(np.arange(0, 1.2, 0.2))
                axes[i, j].plot([0, 1], [0, 1], c='red')

    def show_band_scatterplot(self, band_src_im, band_tgt_im):
        # TODO complete this function
        from scipy.stats import gaussian_kde
        from matplotlib import pyplot as plt

        x = self.im_X.data[band_src_im].flatten()[:10000]
        y = self.im_Y.data[band_tgt_im].flatten()[:10000]

        # Calculate the point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        plt.figure(figsize=(15, 15))
        plt.scatter(x, y, c=z, s=30, edgecolor='none')
        plt.show()


class RefCube(object):
    """Data model class for reference cubes holding the training data for later fitted machine learning classifiers."""

    def __init__(self, filepath='', satellite='', sensor='', LayerBandsAssignment=None):
        # type: (str, str, str, list) -> None
        """Get instance of RefCube.

        :param filepath:                file path for importing an existing reference cube from disk
        :param satellite:               the satellite for which the reference cube holds its spectral data
        :param sensor:                  the sensor for which the reference cube holds its spectral data
        :param LayerBandsAssignment:    the LayerBandsAssignment for which the reference cube holds its spectral data
        """
        # privates
        self._col_imName_dict = dict()
        self._wavelenths = []

        # defaults
        self.data = GeoArray(np.empty((0, 0, len(LayerBandsAssignment) if LayerBandsAssignment else 0)),
                             nodata=-9999)
        self.srcImNames = []

        # args/ kwargs
        self.filepath = filepath
        self.satellite = satellite
        self.sensor = sensor
        self.LayerBandsAssignment = LayerBandsAssignment or []

        if filepath:
            self.read_data_from_disk(filepath)

        if self.satellite and self.sensor and self.LayerBandsAssignment:
            self._add_bandnames_wavelenghts_to_meta()

    def _add_bandnames_wavelenghts_to_meta(self):
        # set bandnames
        self.data.bandnames = ['Band %s' % b for b in self.LayerBandsAssignment]

        # set wavelengths
        self.data.metadata.band_meta['wavelength'] = self.wavelengths

    @property
    def n_images(self):
        """Return the number training images from which the reference cube contains spectral samples."""
        return self.data.shape[1]

    @property
    def n_signatures(self):
        """Return the number spectral signatures per training image included in the reference cube."""
        return self.data.shape[0]

    @property
    def n_clusters(self):
        """Return the number spectral clusters used for clustering source images for the reference cube."""
        if self.filepath:
            identifier = re.search('refcube__(.*).bsq', os.path.basename(self.filepath)).group(1)
            return int(identifier.split('__')[2].split('nclust')[1])

    @property
    def n_signatures_per_cluster(self):
        if self.n_clusters:
            return self.n_signatures // self.n_clusters

    @property
    def col_imName_dict(self):
        # type: () -> OrderedDict
        """Return an ordered dict containing the file base names of the original training images for each column."""
        return OrderedDict((col, imName) for col, imName in zip(range(self.n_images), self.srcImNames))

    @col_imName_dict.setter
    def col_imName_dict(self, col_imName_dict):
        # type: (dict) -> None
        self._col_imName_dict = col_imName_dict
        self.srcImNames = list(col_imName_dict.values())

    @property
    def wavelengths(self):
        if not self._wavelenths and self.satellite and self.sensor and self.LayerBandsAssignment:
            self._wavelenths = list(RSR(self.satellite, self.sensor,
                                        LayerBandsAssignment=self.LayerBandsAssignment).wvl)

        return self._wavelenths

    @wavelengths.setter
    def wavelengths(self, wavelengths):
        self._wavelenths = wavelengths

    def add_refcube_array(self, refcube_array, src_imnames, LayerBandsAssignment):
        # type: (Union[str, np.ndarray], list, list) -> None
        """Add the given given array to the RefCube instance.

        :param refcube_array:           3D array or file path  of the reference cube to be added
                                        (spectral samples /signatures x training images x spectral bands)
        :param src_imnames:             list of training image file base names from which the given cube received data
        :param LayerBandsAssignment:    LayerBandsAssignment of the spectral bands of the given 3D array
        :return:
        """
        # validation
        assert LayerBandsAssignment == self.LayerBandsAssignment, \
            "%s != %s" % (LayerBandsAssignment, self.LayerBandsAssignment)

        if self.data.size:
            new_cube = np.hstack([self.data, refcube_array])
            self.data = GeoArray(new_cube, nodata=self.data.nodata)
        else:
            self.data = GeoArray(refcube_array, nodata=self.data.nodata)

        self.srcImNames.extend(src_imnames)

    def add_spectra(self, spectra, src_imname, LayerBandsAssignment):
        # type: (np.ndarray, str, list) -> None
        """Add a set of spectral signatures to the reference cube.

        :param spectra:              2D numpy array with rows: spectral samples / columns: spectral information (bands)
        :param src_imname:           image basename of the source hyperspectral image
        :param LayerBandsAssignment: LayerBandsAssignment for the spectral dimension of the passed spectra,
                                     e.g., ['1', '2', '3', '4', '5', '6L', '6H', '7', '8']
        """
        # validation
        assert LayerBandsAssignment == self.LayerBandsAssignment, \
            "%s != %s" % (LayerBandsAssignment, self.LayerBandsAssignment)

        # reshape 2D spectra array to one image column (refcube is an image with spectral information in the 3rd dim.)
        im_col = spectra.reshape(spectra.shape[0], 1, spectra.shape[1])

        meta = self.data.metadata  # needs to be copied to the new GeoArray

        if self.data.size:
            # validation
            if spectra.shape[0] != self.data.shape[0]:
                raise ValueError('The number of signatures in the given spectra array does not match the dimensions of '
                                 'the reference cube.')

            # append spectra to existing reference cube
            new_cube = np.hstack([self.data, im_col])
            self.data = GeoArray(new_cube, nodata=self.data.nodata)

        else:
            self.data = GeoArray(im_col, nodata=self.data.nodata)

        # copy previous metadata to the new GeoArray instance
        self.data.metadata = meta

        # add source image name to list of image names
        self.srcImNames.append(src_imname)

    @property
    def metadata(self):
        """Return an ordered dictionary holding the metadata of the reference cube."""
        attrs2include = ['satellite', 'sensor', 'filepath', 'n_signatures', 'n_images', 'n_clusters',
                         'n_signatures_per_cluster', 'col_imName_dict', 'LayerBandsAssignment', 'wavelengths']
        return OrderedDict((k, getattr(self, k)) for k in attrs2include)

    def get_band_combination(self, tgt_LBA):
        # type: (List[str]) -> GeoArray
        """Get an array according to the bands order given by a target LayerBandsAssignment.

        :param tgt_LBA:     target LayerBandsAssignment
        :return:
        """
        if tgt_LBA != self.LayerBandsAssignment:
            cur_LBA_dict = dict(zip(self.LayerBandsAssignment, range(len(self.LayerBandsAssignment))))
            tgt_bIdxList = [cur_LBA_dict[lr] for lr in tgt_LBA]

            return GeoArray(np.take(self.data, tgt_bIdxList, axis=2), nodata=self.data.nodata)
        else:
            return self.data

    def get_spectra_dataframe(self, tgt_LBA):
        # type: (List[str]) -> DataFrame
        """Return a pandas.DataFrame [sample x band] according to the given LayerBandsAssignment.

        :param tgt_LBA: target LayerBandsAssignment
        :return:
        """
        imdata = self.get_band_combination(tgt_LBA)
        spectra = im2spectra(imdata)
        df = DataFrame(spectra, columns=['B%s' % band for band in tgt_LBA])

        return df

    def rearrange_layers(self, tgt_LBA):
        # type: (List[str]) -> None
        """Rearrange the spectral bands of the reference cube according to the given LayerBandsAssignment.

        :param tgt_LBA:     target LayerBandsAssignment
        """
        self.data = self.get_band_combination(tgt_LBA)
        self.LayerBandsAssignment = tgt_LBA

    def read_data_from_disk(self, filepath):
        self.data = GeoArray(filepath)

        with open(os.path.splitext(filepath)[0] + '.meta', 'r') as metaF:
            meta = json.load(metaF)
            for k, v in meta.items():
                if k in ['n_signatures', 'n_images', 'n_clusters', 'n_signatures_per_cluster']:
                    continue  # skip pure getters
                else:
                    setattr(self, k, v)

    def save(self, path_out, fmt='ENVI'):
        # type: (str, str) -> None
        """Save the reference cube to disk.

        :param path_out:    output path on disk
        :param fmt:         output format as GDAL format code
        :return:
        """
        self.filepath = self.filepath or path_out
        self.data.save(out_path=path_out, fmt=fmt)

        # save metadata as JSON file
        with open(os.path.splitext(path_out)[0] + '.meta', 'w') as metaF:
            json.dump(self.metadata.copy(), metaF, separators=(',', ': '), indent=4)

    def _get_spectra_by_label_imname(self, cluster_label, image_basename, n_spectra2get=100, random_state=0):
        cluster_start_pos_all = list(range(0, self.n_signatures, self.n_signatures_per_cluster))
        cluster_start_pos = cluster_start_pos_all[cluster_label]
        spectra = self.data[cluster_start_pos: cluster_start_pos + self.n_signatures_per_cluster,
                            self.srcImNames.index(image_basename)]
        idxs_specIncl = np.random.RandomState(seed=random_state).choice(range(self.n_signatures_per_cluster),
                                                                        n_spectra2get)
        return spectra[idxs_specIncl, :]

    def plot_sample_spectra(self, image_basename, cluster_label='all', include_mean_spectrum=True,
                            include_median_spectrum=True, ncols=5, **kw_fig):
        # type: (Union[str, int, List], str, bool, bool, int, dict) -> plt.figure
        from matplotlib import pyplot as plt

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
            spectra = self._get_spectra_by_label_imname(cluster_label, image_basename, 100)
            for i in range(100):
                plt.plot(self.wavelengths, spectra[i, :])

            plt.xlabel('wavelength [nm]')
            plt.ylabel('%s %s\nreflectance [0-10000]' % (self.satellite, self.sensor))
            plt.title('Cluster #%s' % cluster_label)

            if include_mean_spectrum:
                plt.plot(self.wavelengths, np.mean(spectra, axis=0), c='black', lw=3)
            if include_median_spectrum:
                plt.plot(self.wavelengths, np.median(spectra, axis=0), '--', c='black', lw=3)

        # create a plot with multiple subplots
        else:
            nplots = len(lbls2plot)
            ncols = nplots if nplots < ncols else ncols
            nrows = nplots // ncols if not nplots % ncols else nplots // ncols + 1
            figsize = (4 * ncols, 3 * nrows)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex='all', sharey='all',
                                     **kw_fig)

            for lbl, ax in tqdm(zip(lbls2plot, axes.flatten()), total=nplots):
                spectra = self._get_spectra_by_label_imname(lbl, image_basename, 100)

                for i in range(100):
                    ax.plot(self.wavelengths, spectra[i, :], lw=1)

                if include_mean_spectrum:
                    ax.plot(self.wavelengths, np.mean(spectra, axis=0), c='black', lw=2)
                if include_median_spectrum:
                    ax.plot(self.wavelengths, np.median(spectra, axis=0), '--', c='black', lw=3)

                ax.grid(lw=0.2)
                ax.set_ylim(0, 10000)

                if ax.is_last_row():
                    ax.set_xlabel('wavelength [nm]')
                if ax.is_first_col():
                    ax.set_ylabel('%s %s\nreflectance [0-10000]' % (self.satellite, self.sensor))
                ax.set_title('Cluster #%s' % lbl)

        fig.suptitle("Refcube spectra from image '%s':" % image_basename, fontsize=15)
        plt.tight_layout(rect=(0, 0, 1, .95))
        plt.show()

        return fig
