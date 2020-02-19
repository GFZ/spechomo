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
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

from glob import glob
import os
from zipfile import ZipFile
import dill
from multiprocessing import Pool
from typing import Union, List  # noqa F401  # flake8 issue

import numpy as np  # noqa F401  # flake8 issue
from geoarray import GeoArray  # noqa F401  # flake8 issue
import pandas as pd
from tempfile import TemporaryDirectory
from natsort import natsorted

from .options import options


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


_columns_df_trafos = ['method', 'src_sat', 'src_sen', 'src_LBA', 'tgt_sat', 'tgt_sen', 'tgt_LBA', 'n_clusters']


def list_available_transformations(classifier_rootDir=options['classifiers']['rootdir'],
                                   method=None,
                                   src_sat=None, src_sen=None, src_LBA=None,
                                   tgt_sat=None, tgt_sen=None, tgt_LBA=None,
                                   n_clusters=None):
    # type: (str, str, str, str, List[str], str, str, List[str], int) -> pd.DataFrame
    """List all sensor transformations available according to the given classifier root directory.

    NOTE:   This function can be used to copy/paste possible input parameters for
            spechomo.SpectralHomogenizer.predict_by_machine_learner().

    :param classifier_rootDir:  directory containing classifiers for homogenization, either as .zip archives or
                                as .dill files
    :param method:              filter results by the machine learning approach to be used for spectral bands prediction
    :param src_sat:             filter results by source satellite, e.g., 'Landsat-8'
    :param src_sen:             filter results by source sensor, e.g., 'OLI_TIRS'
    :param src_LBA:             filter results by source bands list
    :param tgt_sat:             filter results by target satellite, e.g., 'Landsat-8'
    :param tgt_sen:             filter results by target sensor, e.g., 'OLI_TIRS'
    :param tgt_LBA:             filter results by target bands list
    :param n_clusters:          filter results by the number of spectral clusters to be used during LR/ RR/ QR
                                homogenization
    :return:    pandas.DataFrame listing all the available transformations
    """
    clf_zipfiles = glob(os.path.join(classifier_rootDir, '*_classifiers.zip'))

    df = pd.DataFrame(columns=_columns_df_trafos)

    if clf_zipfiles:
        for path_zipfile in clf_zipfiles:

            with TemporaryDirectory() as td, ZipFile(path_zipfile) as zF:
                zF.extractall(td)

                with Pool() as pool:
                    paths_dillfiles = [(os.path.join(td, fN)) for fN in natsorted(zF.namelist())]
                    dfs = pool.map(explore_classifer_dillfile, paths_dillfiles)

                df = pd.concat([df] + dfs)

    else:
        paths_dillfiles = natsorted(glob(os.path.join(classifier_rootDir, '*.dill')))

        if paths_dillfiles:
            with Pool() as pool:
                dfs = pool.map(explore_classifer_dillfile, paths_dillfiles)

            df = pd.concat([df] + dfs)

    # apply filters
    if not df.empty:
        df = df.reset_index(drop=True)

        def _force_list(arg):
            return [arg] if not isinstance(arg, list) else arg

        for filterparam in _columns_df_trafos:
            if locals()[filterparam]:
                df = df[df[filterparam].isin(_force_list(locals()[filterparam]))]

    return df


def explore_classifer_dillfile(path_dillFile):
    # type: (str) -> pd.DataFrame
    """List all homogenization transformations included in the given .dill file.

    :param path_dillFile:
    :return:
    """
    df = pd.DataFrame(columns=_columns_df_trafos)

    meth_nclust_str, src_sat, src_sen = os.path.splitext(os.path.basename(path_dillFile))[0].split('__')
    method, nclust_str = meth_nclust_str.split('_')
    nclust = int(nclust_str.split('clust')[1])

    with open(path_dillFile, 'rb') as inF:
        content = dill.load(inF)

        for src_LBA, subdict_L1 in content.items():
            for tgt_sat_sen, subdict_L2 in subdict_L1.items():
                tgt_sat, tgt_sen = tgt_sat_sen

                for tgt_LBA, subdict_L3 in subdict_L2.items():
                    df.loc[len(df)] = [method,
                                       src_sat, src_sen, src_LBA.split('__'),
                                       tgt_sat, tgt_sen, tgt_LBA.split('__'),
                                       nclust]

    return df
