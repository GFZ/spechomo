# -*- coding: utf-8 -*-
"""Unit test package for spechomo."""

import os
from gms_preprocessing import set_config

gms_db_host = 'localhost' if 'GMS_db_host' not in os.environ else os.environ['GMS_db_host']  # FIXME remove
gms_config = set_config(job_ID=26186196, db_host=gms_db_host, reset_status=True)  # FIXME remove
