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


class ClassifierNotAvailableError(RuntimeError):
    def __init__(self, spechomo_method, src_sat, src_sen, src_LBA, tgt_sat, tgt_sen, tgt_LBA, n_clusters):
        self.spechomo_method = spechomo_method
        self.src_sat = src_sat
        self.src_sen = src_sen
        self.src_LBA = src_LBA
        self.tgt_sat = tgt_sat
        self.tgt_sen = tgt_sen
        self.tgt_LBA = tgt_LBA
        self.n_clusters = n_clusters
        RuntimeError.__init__(self)

    def __str__(self):
        return 'No %s classifier available for predicting %s %s %s from %s %s %s (%d clusters).'\
               % (self.spechomo_method, self.tgt_sat, self.tgt_sen, self.tgt_LBA,
                  self.src_sat, self.src_sen, self.src_LBA, self.n_clusters)
