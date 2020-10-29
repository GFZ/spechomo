============
Installation
============

Using Anaconda or Miniconda (recommended)
-----------------------------------------

Using conda_ (latest version recommended), SpecHomo is installed as follows:


1. Create virtual environment for SpecHomo (optional but recommended):

   .. code-block:: bash

    $ conda create -c conda-forge --name spechomo python=3
    $ conda activate spechomo


2. Then install spechomo itself:

   .. code-block:: bash

    $ conda install -c conda-forge spechomo


This is the preferred method to install SpecHomo, as it always installs the most recent stable release and
automatically resolves all the dependencies.


Using pip (not recommended)
---------------------------

There is also a `pip`_ installer for SpecHomo. However, please note that SpecHomo depends on some
open source packages that may cause problems when installed with pip. Therefore, we strongly recommend
to resolve the following dependencies before the pip installer is run:

    * cartopy
    * gdal
    * geopandas
    * matplotlib
    * numpy
    * pandas
    * pyproj
    * scikit-learn >=0.23.2
    * scikit-image
    * shapely

Then, the pip installer can be run by:

   .. code-block:: bash

    $ pip install spechomo

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.



.. note::

    The SpecHomo package has been tested with Python 3.4+ and Python 2.7. It should be fully compatible to all Python
    versions from 2.7 onwards. However, we will continously drop the support for Python 2.7 in future.


.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _conda: https://conda.io/docs
.. _`dependencies of SpecHomo`: https://gitext.gfz-potsdam.de/danschef/arosics/-/blob/master/requirements.txt
