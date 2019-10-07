Which sensor transformations are available?
-------------------------------------------

SpecHomo enables you to perform spectral harmonization for various sensors - in different combinations and with
different band compositions. To get a list of all the available transformations, you may use the function
:func:`list_available_transformations<spechomo.utils.list_available_transformations>`.

.. note::

    You can directly copy/paste the outputs of this function to the method you need for running the homogenization:
    :meth:`spechomo.prediction.SpectralHomogenizer.predict_by_machine_learner()
    <spechomo.prediction.SpectralHomogenizer.predict_by_machine_learner>`. The column names of the returned
    table are consistent with this method.


Here is the code to get the full list of available transformations:

.. code-block:: python

    >>> from spechomo.utils import list_available_transformations
    >>> list_available_transformations()

.. image:: ../images/list_available_transformations_no_filters.png


The output may also be directly filtered using the input parameters of the function. Here is an example:

.. code-block:: python

    >>> from spechomo.utils import list_available_transformations
    >>> list_available_transformations(method='LR', src_sat='Landsat-8', n_clusters=50)

.. image:: ../images/list_available_transformations_filtered.png


.. note::

    If you have trained your own classifiers for homogenization, you may also pass a different classifier root
    directory to :func:`list_available_transformations<spechomo.utils.list_available_transformations>`. Use the
    parameter `classifier_rootDir` for this.
