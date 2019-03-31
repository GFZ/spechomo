#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from spechomo import __path__
from spechomo.classifier_creation import ReferenceCube_Generator, RefCube, ClusterClassifier_Generator

from gms_preprocessing.model.gms_object import GMS_identifier  # FIXME

hyspec_data = os.path.join(__path__[0], '../tests/data/Bavaria_farmland_LMU_Hyspex_subset.bsq')
refcube_l8 = os.path.join(__path__[0], '../tests/data/refcube__Landsat-8__OLI_TIRS__nclust50__nsamp100.bsq')
refcube_l5 = os.path.join(__path__[0], '../tests/data/refcube__Landsat-5__TM__nclust50__nsamp100.bsq')


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
                                          v=False)

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

        from gms_preprocessing.io.input_reader import SRF
        tgt_srf = SRF(GMS_identifier(satellite='Sentinel-2A', sensor='MSI', subsystem='', image_type='RSD',
                                     dataset_ID=-9999, proc_level='L1A', logger=None))
        unif_random_spectra_rsp = \
            self.SHC.resample_spectra(unif_random_spectra,
                                      src_cwl=np.array(src_im.meta.band_meta['wavelength'], dtype=np.float).flatten(),
                                      tgt_srf=tgt_srf)
        self.assertIsInstance(unif_random_spectra_rsp, np.ndarray)
        self.assertEqual(unif_random_spectra_rsp.shape, (self.tgt_n_samples, len(tgt_srf.bands)))

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
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_l5])
        CCG.create_classifiers(outDir=self.tmpOutdir.name, method='LR', n_clusters=5)

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
        CCG = ClusterClassifier_Generator([refcube_l8, refcube_l5])
        CCG.create_classifiers(outDir=self.tmpOutdir.name, method='RFR', n_clusters=1,
                               **dict(n_jobs=-1, n_estimators=20, max_depth=10))

        outpath_cls = os.path.join(self.tmpOutdir.name,
                                   'RFR_trees%d_clust1__Landsat-8__OLI_TIRS.dill' % 20)
        self.assertTrue(os.path.exists(outpath_cls))

        with open(outpath_cls, 'rb') as inF:
            undilled = dill.load(inF)
            self.assertIsInstance(undilled, dict)
            self.assertTrue(bool(undilled), msg='Generated classifier collection is empty.')
