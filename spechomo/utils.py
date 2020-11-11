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

from glob import glob
import os
from zipfile import ZipFile
import dill
from multiprocessing import Pool
from typing import Union, List  # noqa F401  # flake8 issue
import json
import warnings
from urllib.request import urlretrieve, urlopen

import numpy as np  # noqa F401  # flake8 issue
from geoarray import GeoArray  # noqa F401  # flake8 issue
import pandas as pd
from tempfile import TemporaryDirectory
from natsort import natsorted
from tqdm import tqdm

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
    return spectra.reshape((tgt_rows, tgt_cols, spectra.shape[1]))


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


def export_classifiers_as_JSON(export_rootDir,
                               classifier_rootDir=options['classifiers']['rootdir'],
                               method=None,
                               src_sat=None, src_sen=None, src_LBA=None,
                               tgt_sat=None, tgt_sen=None, tgt_LBA=None,
                               n_clusters=None):
    # type: (str, str, str, str, str, List[str], str, str, List[str], int) -> None
    """Export spectral harmonization classifiers as JSON files that match the provided filtering criteria.

    NOTE: So far, this function will only work for LR classifiers.

    :param export_rootDir:      directory where to save the exported JSON files
    :param classifier_rootDir:  directory containing classifiers for homogenization, either as .zip archives or
                                as .dill files
    :param method:              filter by the machine learning approach to be used for spectral bands prediction
    :param src_sat:             filter by source satellite, e.g., 'Landsat-8'
    :param src_sen:             filter by source sensor, e.g., 'OLI_TIRS'
    :param src_LBA:             filter by source bands list
    :param tgt_sat:             filter by target satellite, e.g., 'Landsat-8'
    :param tgt_sen:             filter by target sensor, e.g., 'OLI_TIRS'
    :param tgt_LBA:             filter by target bands list
    :param n_clusters:          filter by the number of spectral clusters to be used during LR/ RR/ QR
                                homogenization
    :return:
    """
    from .classifier import Cluster_Learner

    # get matching classifiers
    df_trafos = list_available_transformations(classifier_rootDir=classifier_rootDir,
                                               method=method,
                                               src_sat=src_sat, src_sen=src_sen, src_LBA=src_LBA,
                                               tgt_sat=tgt_sat, tgt_sen=tgt_sen, tgt_LBA=tgt_LBA,
                                               n_clusters=n_clusters)

    if len(df_trafos):
        # export each matching transformation to a JSON file
        for i, trafo in tqdm(df_trafos.iterrows(), total=len(df_trafos)):

            CL = Cluster_Learner.from_disk(
                classifier_rootDir=classifier_rootDir,
                method=trafo.method,
                n_clusters=trafo.n_clusters,
                src_satellite=trafo.src_sat, src_sensor=trafo.src_sen, src_LBA=trafo.src_LBA,
                tgt_satellite=trafo.tgt_sat, tgt_sensor=trafo.tgt_sen, tgt_LBA=trafo.tgt_LBA)

            path_out = os.path.join(export_rootDir,
                                    trafo.method,
                                    "src__%s__%s__%s" % (trafo.src_sat, trafo.src_sen, '_'.join(trafo.src_LBA)),
                                    "to__%s__%s__%s" % (trafo.tgt_sat, trafo.tgt_sen, '_'.join(trafo.tgt_LBA)),
                                    "%s__clust%d.json" % (trafo.method, trafo.n_clusters)
                                    )
            os.makedirs(os.path.dirname(path_out), exist_ok=True)

            with open(path_out, 'w') as outF:
                json.dump(CL.to_jsonable_dict(), outF, indent=4, sort_keys=True)

    else:
        warnings.warn('No classifiers found matching the provided filter criteria. Nothing exported.', RuntimeWarning)


def download_pretrained_classifiers(method, tgt_dir=options['classifiers']['rootdir']):
    remote_filespecs = {
        '100k_conservrsp_SCA_SD100percSA90perc_without_aviris__SCADist90pSAM40p': {
            # 'LR': 'https://nextcloud.gfz-potsdam.de/s/Rzb75kckBreFfNE/download',  # 20201008
            'LR': 'https://nextcloud.gfz-potsdam.de/s/mZEnS5g7AGWyRHB/download',
            # 'QR': 'https://nextcloud.gfz-potsdam.de/s/Kk4zoCXxAEkAFZL/download',  # 20201008
            'QR': 'https://nextcloud.gfz-potsdam.de/s/JcQDbZBtTiw9NYi/download',
        }
    }
    clf_name = options['classifiers']['name']
    try:
        url = remote_filespecs[clf_name][method]
    except KeyError:
        raise RuntimeError("Currently there are no %s classifiers named '%s' available." % (method, clf_name))

    for i in range(3):  # try 3 times
        if i > 0:
            print('Download failed. Restarting...')

        # get filename
        fn = urlopen(url).headers['content-disposition'].split('filename=')[-1].replace('"', '')

        # download
        if not os.path.isdir(tgt_dir):
            os.makedirs(tgt_dir)
        outP, msg = urlretrieve(url, os.path.join(tgt_dir, fn))

        if os.path.getsize(outP) == int(msg.get('content-length')):
            return outP
