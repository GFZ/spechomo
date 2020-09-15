Predicting spectral information / multi-sensor homogenization
-------------------------------------------------------------

To execute the spectral homogenization, i.e., to transform the spectral information from one sensor into the spectral
domain of another one, SpecHomo provides the :class:`SpectralHomogenizer<spechomo.prediction.SpectralHomogenizer>`
class. Please see the linked content for a full documentation of this class.

For the sake of simplicity, the usage of this class is described below, at the
**example of Landsat-8 data, spectrally adapted to Sentinel-2A**. Transformations between various other sensors are
possible, see `here <https://geomultisens.gitext-pages.gfz-potsdam.de/spechomo/doc/usage/available_transformations.html
#which-sensor-transformations-are-available>`__.

First, load the Landsat-8 `surface reflectance`_ image that you want to transform to the spectral domain of Sentinel-2A
(we use the `geoarray`_ library for this - it is installed with SpecHomo):

.. code-block:: python

    from geoarray import GeoArray

    image_l8 = GeoArray('/path/to/your/Landsat/image/LC81940242014072LGN00_surface_reflectance__stacked.bsq')

.. attention::

    Please make sure, that the Landsat-8 input image contains the right bands in the correct order before you run the
    homogenization! By running the :func:`list_available_transformations<spechomo.utils.list_available_transformations>`
    function as described
    `here <https://geomultisens.gitext-pages.gfz-potsdam.de/spechomo/doc/usage/available_transformations.html>`__, you can
    find out, that the needed band list is ['1', '2', '3', '4', '5', '6', '7']. These band numbers refer to the
    official provider band-names as described for Landsat at the
    `USGS website <https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites>`__.


Now run the homogenization by using the :class:`SpectralHomogenizer<spechomo.prediction.SpectralHomogenizer>` class as
follows:

.. code-block:: python

    from spechomo import SpectralHomogenizer

    # get an instance of SpectralHomogenizer class:
    SH = SpectralHomogenizer()

    # run the spectral homogenization
    image_s2, errors = SH.predict_by_machine_learner(
        arrcube=image_l8[:,:,:7],
        method='LR',
        n_clusters=50,
        src_satellite='Landsat-8',
        src_sensor='OLI_TIRS',
        src_LBA=['1', '2', '3', '4', '5', '6', '7'],  # must be passed as list of strings and match the band numbers of the input image
        tgt_satellite='Sentinel-2A',
        tgt_sensor='MSI',
        tgt_LBA=['1', '2', '3', '4', '5', '6', '7', '8', '8A', '11', '12'],
        classif_alg='kNN_SAM',
        global_clf_threshold=4
    )

    # save the Sentinel-2A adapted Landsat-8 image to disk
    image_s2.save('/your/output/path/l8_s2_homogenization_result.bsq')

    # save the estimated homogenization errors/uncertainties to disk
    errors.save('/your/output/path/l8_s2_homogenization_errors.bsq')

.. note::

    * You can directly copy/paste possible input parameters for the
      :meth:`predict_by_machine_learner<spechomo.prediction.SpectralHomogenizer.predict_by_machine_learner>` method
      from the :func:`list_available_transformations<spechomo.utils.list_available_transformations>`
      function as described
      `here <https://geomultisens.gitext-pages.gfz-potsdam.de/spechomo/doc/usage/available_transformations.html
      #which-sensor-transformations-are-available>`__.
    * You may also save the homogenization results to other GDAL compatible image formats
      (see :meth:`geoarray.GeoArray.save` for details).
    * Further explanation on input parameters like `method`, `n_clusters`, `classif_alg` or `global_clf_threshold` is
      given in the :meth:`predict_by_machine_learner<spechomo.prediction.SpectralHomogenizer.predict_by_machine_learner>`
      method documentation. See `Scheffler et al. 2020 <https://doi.org/10.1016/j.rse.2020.111723>`__ for a thorough
      evaluation of the different homogenization algorithms available in the SpecHomo library.


.. _`surface reflectance`: https://geomultisens.gitext-pages.gfz-potsdam.de/spechomo/doc/usage/input_data_requirements.html#surface-reflectance
.. _`geoarray`: https://gitext.gfz-potsdam.de/danschef/geoarray
