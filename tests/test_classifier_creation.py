#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spechomo, Spectral homogenization of multispectral satellite data
#
# Copyright (C) 2019-2021
# - Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz-potsdam.de)
# - Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam,
#   Germany (https://www.gfz-potsdam.de/)
#
# This software was developed within the context of the GeoMultiSens project funded
# by the German Federal Ministry of Education and Research
# (project grant code: 01 IS 14 010 A-C).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Please note the following exception: `spechomo` depends on tqdm, which is
# distributed under the Mozilla Public Licence (MPL) v2.0 except for the files
# "tqdm/_tqdm.py", "setup.py", "README.rst", "MANIFEST.in" and ".gitignore".
# Details can be found here: https://github.com/tqdm/tqdm/blob/master/LICENCE.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
test_classifier_creation
------------------------

Tests for spechomo.classifier_creation
"""

import unittest
import os
import tempfile
import numpy as np
import dill
from geoarray import GeoArray
from pyrsr import RSR

from spechomo import __path__
from spechomo.classifier_creation import ReferenceCube_Generator, RefCube, ClusterClassifier_Generator

hyspec_data = os.path.join(__path__[0], '../tests/data/AV_mastercal_testdata.bsq')
refcube_l8 = os.path.join(__path__[0], '../tests/data/refcube__Landsat-8__OLI_TIRS__nclust50__nsamp1000.bsq')
refcube_l5 = os.path.join(__path__[0], '../tests/data/refcube__Landsat-5__TM__nclust50__nsamp1000.bsq')
refcube_s2 = os.path.join(__path__[0], '../tests/data/refcube__Sentinel-2A__MSI__nclust50__nsamp1000.bsq')


class Test_ReferenceCube_Generator(unittest.TestCase):
    """Tests class for spechomo.classifier_creation.L2B_P.ReferenceCube_Generator"""

    @classmethod
    def setUpClass(cls):
        cls.tmpOutdir = tempfile.TemporaryDirectory()
        cls.testIms = [hyspec_data, hyspec_data, ]
        cls.tgt_sat_sen_list = [
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
        cls.n_clusters = 5
        cls.tgt_n_samples = 500
        cls.SHC = ReferenceCube_Generator(cls.testIms, dir_refcubes=cls.tmpOutdir.name,
                                          tgt_sat_sen_list=cls.tgt_sat_sen_list,
                                          n_clusters=cls.n_clusters,
                                          tgt_n_samples=cls.tgt_n_samples,
                                          dir_clf_dump=cls.tmpOutdir.name,
                                          v=False,
                                          CPUs=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmpOutdir.cleanup()

    def test_cluster_image_and_get_uniform_samples(self):
        src_im = self.SHC.ims_ref[0]
        baseN = os.path.splitext(os.path.basename(src_im))[0]
        unif_random_spectra = self.SHC.cluster_image_and_get_uniform_spectra(src_im, basename_clf_dump=baseN)
        self.assertIsInstance(unif_random_spectra, np.ndarray)
        self.assertEqual(unif_random_spectra.shape, (self.tgt_n_samples, GeoArray(src_im).bands))

    def test_resample_spectra(self):
        src_im = GeoArray(self.SHC.ims_ref[0])
        unif_random_spectra = self.SHC.cluster_image_and_get_uniform_spectra(src_im)

        tgt_rsr = RSR(satellite='Sentinel-2A', sensor='MSI')
        # tgt_rsr = RSR(satellite='Terra', sensor='MODIS', sort_by_cwl=True)
        unif_random_spectra_rsp = \
            self.SHC.resample_spectra(unif_random_spectra,
                                      src_cwl=np.array(src_im.meta.band_meta['wavelength'], dtype=float).flatten(),
                                      tgt_rsr=tgt_rsr,
                                      nodataVal=src_im.nodata)
        self.assertIsInstance(unif_random_spectra_rsp, np.ndarray)
        self.assertEqual(unif_random_spectra_rsp.shape, (self.tgt_n_samples, len(tgt_rsr.bands)))

    def test_generate_reference_cube(self):
        refcubes = self.SHC.generate_reference_cubes()
        self.assertIsInstance(refcubes, dict)
        self.assertIsInstance(refcubes[('Landsat-8', 'OLI_TIRS')], RefCube)
        self.assertEqual(refcubes[('Landsat-8', 'OLI_TIRS')].data.shape, (self.tgt_n_samples, len(self.testIms), 7))
        self.assertNotEqual(len(os.listdir(self.tmpOutdir.name)), 0)

    # @unittest.SkipTest
    # def test_multiprocessing(self):
    #     SHC = ReferenceCube_Generator_OLD([testdata, testdata, ], v=False, CPUs=1)
    #     ref_cube_sp = SHC.generate_reference_cube('Landsat-8', 'OLI_TIRS', n_clusters=10, tgt_n_samples=1000)
    #
    #     SHC = ReferenceCube_Generator_OLD([testdata, testdata, ], v=False, CPUs=None)
    #     ref_cube_mp = SHC.generate_reference_cube('Landsat-8', 'OLI_TIRS', n_clusters=10, tgt_n_samples=1000)
    #
    #     self.assertTrue(np.any(ref_cube_sp), msg='Singleprocessing result is empty.')
    #     self.assertTrue(np.any(ref_cube_mp), msg='Multiprocessing result is empty.')


class Test_ClusterClassifier_Generator(unittest.TestCase):
    """Tests class for spechomo.classifier_creation.Classifier_Generator"""

    @classmethod
    def setUpClass(cls):
        cls.tmpOutdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmpOutdir.cleanup()

    def test_init_from_path_strings(self):
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_l5])
        self.assertIsInstance(CCG, ClusterClassifier_Generator)

    def test_init_from_RefCubes(self):
        RC = RefCube(refcube_l8)
        CCG = ClusterClassifier_Generator([RC, RC])
        self.assertIsInstance(CCG, ClusterClassifier_Generator)

    def test_create_classifiers_LR(self):
        """Test creation of linear regression classifiers."""
        # CCG = ClusterClassifier_Generator([refcube_l8, refcube_l5])
        # CCG = ClusterClassifier_Generator([refcube_l7, refcube_s2])
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_s2])
        # CCG = ClusterClassifier_Generator([refcube_s2, refcube_l8, ])
        CCG.create_classifiers(outDir=self.tmpOutdir.name, method='LR', n_clusters=5,
                               max_distance='10%', max_angle=3)

        outpath_cls = os.path.join(self.tmpOutdir.name, 'LR_clust5__Landsat-8__OLI_TIRS.dill')
        self.assertTrue(os.path.exists(outpath_cls))

        with open(outpath_cls, 'rb') as inF:
            undilled = dill.load(inF)
            self.assertIsInstance(undilled, dict)
            self.assertTrue(bool(undilled), msg='Generated classifier collection is empty.')

    def test_create_classifiers_RR(self):
        """Test creation of ridge regression classifiers."""
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_l5])
        CCG.create_classifiers(outDir=self.tmpOutdir.name, method='RR', n_clusters=5)

        outpath_cls = os.path.join(self.tmpOutdir.name, 'RR_alpha1.0_clust5__Landsat-8__OLI_TIRS.dill')
        self.assertTrue(os.path.exists(outpath_cls))

        with open(outpath_cls, 'rb') as inF:
            undilled = dill.load(inF)
            self.assertIsInstance(undilled, dict)
            self.assertTrue(bool(undilled), msg='Generated classifier collection is empty.')

    def test_create_classifiers_QR(self):
        """Test creation of quadratic regression classifiers."""
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_l5])
        CCG.create_classifiers(outDir=self.tmpOutdir.name, method='QR', n_clusters=5)

        outpath_cls = os.path.join(self.tmpOutdir.name, 'QR_clust5__Landsat-8__OLI_TIRS.dill')
        self.assertTrue(os.path.exists(outpath_cls))

        with open(outpath_cls, 'rb') as inF:
            undilled = dill.load(inF)
            self.assertIsInstance(undilled, dict)
            self.assertTrue(bool(undilled), msg='Generated classifier collection is empty.')

    def test_create_classifiers_RFR(self):
        """Test creation of random forest regression classifiers."""
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_s2])
        CCG.create_classifiers(outDir=self.tmpOutdir.name, method='RFR', n_clusters=1,
                               **dict(n_jobs=-1, n_estimators=20, max_depth=10))

        outpath_cls = os.path.join(self.tmpOutdir.name,
                                   'RFR_trees%d_clust1__Landsat-8__OLI_TIRS.dill' % 20)
        self.assertTrue(os.path.exists(outpath_cls))

        with open(outpath_cls, 'rb') as inF:
            undilled = dill.load(inF)
            self.assertIsInstance(undilled, dict)
            self.assertTrue(bool(undilled), msg='Generated classifier collection is empty.')

    def test_preview_classifiers(self):
        method = 'LR'
        n_clusters = 5
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_s2])
        # CCG.create_classifiers(outDir=self.tmpOutdir.name,
        #                        method='LR',
        #                        n_clusters=1,
        #                        CPUs=32,
        #                        sam_classassignment=True,
        #                        max_distance=max_distance,
        #                        max_angle=max_angle)
        CCG.create_classifiers(outDir=self.tmpOutdir.name,
                               method=method,
                               n_clusters=n_clusters,
                               CPUs=32,
                               sam_classassignment=True,
                               max_distance='10%',
                               max_angle=3)

        from spechomo.classifier import Cluster_Learner, ClassifierCollection
        coll = ClassifierCollection(
            os.path.join(self.tmpOutdir.name, '%s_clust%s__Landsat-8__OLI_TIRS.dill' % (method, n_clusters)))
        CL = Cluster_Learner(
            coll['1__2__3__4__5__6__7'][('Sentinel-2A', 'MSI')]['1__2__3__4__5__6__7__8__8A__11__12'],
            global_classifier=None)
        fig, axes = CL.plot_sample_spectra(dpi=60, ncols=6)


if __name__ == '__main__':
    import pytest
    pytest.main()
