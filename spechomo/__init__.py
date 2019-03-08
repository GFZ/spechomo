# -*- coding: utf-8 -*-

import os
from .version import __version__, __versionalias__   # noqa (E402 + F401)
from gms_preprocessing import set_config


__author__ = """Daniel Scheffler"""
__email__ = 'daniel.scheffler@gfz-potsdam.de'

gms_db_host = 'localhost' if 'GMS_db_host' not in os.environ else os.environ['GMS_db_host']  # FIXME remove
gms_config = set_config(job_ID=26186196, db_host=gms_db_host, reset_status=True)  # FIXME remove
