# -*- coding: utf-8 -*-

import os
import yaml

from spechomo import __path__


with open(os.path.join(__path__[0], 'options_default.yaml'), 'r') as stream:
    options = yaml.load(stream)
