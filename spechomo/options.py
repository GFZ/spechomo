# -*- coding: utf-8 -*-

import os
import yaml

from spechomo import __path__ as spechomo_rootdir


with open(os.path.join(spechomo_rootdir, 'options_default.yaml'), 'r') as stream:
    options = yaml.load(stream)
