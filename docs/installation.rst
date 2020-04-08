============
Installation
============

The spechomo package depends on some open source packages which are usually installed without problems by the automatic
install routine. However, for some projects, we strongly recommend resolving dependencies before the automatic
installer is run. This approach avoids problems with conflicting versions of the same software.
Using conda_, the recommended approach is:


1. Create virtual environment for spechomo (optional but recommended):

   .. code-block:: bash

    $ conda create -c conda-forge --name spechomo python=3
    $ source activate spechomo


2. Install some libraries needed for spechomo:

   .. code-block:: bash

    $ conda install -c conda-forge matplotlib numpy pandas "scikit-learn=0.19.1" basemap gdal "geopandas<0.6.3" pyproj scikit-image shapely


3. Then install spechomo using the pip installer:

   .. code-block:: bash

    $ pip install git+https://gitext.gfz-potsdam.de/geomultisens/spechomo.git


This is the preferred method to install spechomo, as it will always install the most recent stable release.

.. note::

    The spechomo package has been tested with Python 3.4+ and Python 2.7. It should be fully compatible to all Python
    versions from 2.7 onwards. However, we will continously drop the support for Python 2.7 in future.


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _conda: https://conda.io/docs
