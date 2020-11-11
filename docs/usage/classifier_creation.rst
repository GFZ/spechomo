Create your own classifiers
---------------------------

Although SpecHomo already includes a lot of classifiers for spectral harmonization, it might be useful for some
applications to train your own.

The way to create some SpecHomo-compatible classifiers is described below. Details on the underlying algorithms can be
found in `Scheffler et al. 2020 <https://doi.org/10.1016/j.rse.2020.111723>`__.

1. Build up a spectral database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you need to create a spectral database that can later be used to train classifiers for spectral homogenization.
This database consists on multiple, so-called *reference cubes* - one for each source or target sensor of the spectral
homogenization. To generate these reference cubes, SpecHomo provides the
:class:`ReferenceCube_Generator<spechomo.classifier_creation.ReferenceCube_Generator>` class. Please see the linked
content for a full documentation. As input you need to provide hyperspectral datasets containing those land-cover types
that you want to include into your own classifiers.

**These hyperspectral datasets must fulfil the following requirements:**

* GDAL compatible file format (find a list `here <http://www.gdal.org/formats_list.html>`__)
* surface reflectance, scaled between 1 and 10000 (10000 represents 100% reflectance)
* metadata contain a no data value and wavelength positions of all included bands in nanometers
* high signal-to-noise ratio
* no bad bands (containing only no data)

Here is a **basic example how to generate the reference cubes** for Sentinel-2A and Landsat-8 from three hyperspectral
images:

.. code-block:: python

    from spechomo.classifier_creation import ReferenceCube_Generator

    images_hyperspec = [
        '/path/to/hyperspectral_image1.bsq',
        '/path/to/hyperspectral_image2.bsq',
        '/path/to/hyperspectral_image3.bsq',
    ]

    # get an instance of the ReferenceCube_Generator class
    RCG = ReferenceCube_Generator(images_hyperspec,
                                  dir_refcubes='/path/where/to/save/the/generated/reference_cubes/',
                                  n_clusters=50,
                                  tgt_sat_sen_list=[
                                      ('Sentinel-2A', 'MSI'),
                                      ('Landsat-8', 'OLI_TIRS')
                                  ],
                                  tgt_n_samples=10000)

    # run reference cube generation
    RCG.generate_reference_cubes()  # instead of using the defaults here you may tune a lot of parameters


2. Generate the classifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the generated reference cubes, the classifiers for spectral homogenization can be generated using the
:class:`ClusterClassifier_Generator<spechomo.classifier_creation.ClusterClassifier_Generator>` class. Please see the
linked content for a full documentation.

Here is an example based on the above created reference cubes for Sentinel-2A and Landsat-8:

.. code-block:: python

    from spechomo.classifier_creation import ClusterClassifier_Generator

    refcubes = [
        '/path/to/refcube_Sentinel-2A_MSI.bsq',
        '/path/to/refcube_Landsat-8_OLI_TIRS.bsq',
    ]

    # get an instance of the ClusterClassifier_Generator class
    CCG = ClusterClassifier_Generator(refcubes)

    # run classifier generation
    CCG.create_classifiers(outDir='/path/where/to/save/the/generated/classifiers/',
                           method='LR',
                           n_clusters=50)  # there are also some further parameters to tune here

The classifiers are saved as *.dill files. You may explore them later using the
:func:`list_available_transformations<spechomo.utils.list_available_transformations>` function as described
`here <https://geomultisens.gitext-pages.gfz-potsdam.de/spechomo/doc/usage/available_transformations.html>`__.

.. note::

    You may convert the saved classifiers to JSON format, e.g., for using them in different programming environments.
    To do so, use the :func:`export_classifiers_as_JSON<spechomo.utils.export_classifiers_as_JSON>` function. However,
    this will currently only work for linear regression (LR) classifiers.
