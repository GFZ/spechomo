==================================================================
SpecHomo - Spectral homogenization of multispectral satellite data
==================================================================

* Free software: GNU General Public License v3
* **Documentation:** http://geomultisens.gitext.gfz-potsdam.de/spechomo/doc/
* The (open-access) **paper** corresponding to this software repository has been submitted to
  Remote Sensing of Environment:
  Scheffler D, Segl K, Frantz D. Spectral Adjustment and Red Edge Prediction of Landsat-8 to Sentinel-2 using
  land-cover specific regressors.
* Submit feedback by filing an issue `here <https://gitext.gfz-potsdam.de/geomultisens/spechomo/issues>`__
  or join our chat here: |Gitter|

.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/spechomo/community#
    :alt: https://gitter.im/spechomo/community#

Status
------

.. .. image:: https://img.shields.io/travis/danschef/spechomo.svg
        :target: https://travis-ci.org/danschef/spechomo

.. .. image:: https://readthedocs.org/projects/spechomo/badge/?version=latest
        :target: https://spechomo.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. .. image:: https://pyup.io/repos/github/danschef/spechomo/shield.svg
     :target: https://pyup.io/repos/github/danschef/spechomo/
     :alt: Updates


.. image:: https://gitext.gfz-potsdam.de/geomultisens/spechomo/badges/master/build.svg
        :target: https://gitext.gfz-potsdam.de/geomultisens/spechomo/commits/master
.. image:: https://gitext.gfz-potsdam.de/geomultisens/spechomo/badges/master/coverage.svg
        :target: http://geomultisens.gitext.gfz-potsdam.de/spechomo/coverage/
.. image:: https://img.shields.io/pypi/v/spechomo.svg
        :target: https://pypi.python.org/pypi/spechomo
.. image:: https://img.shields.io/pypi/l/spechomo.svg
        :target: https://gitext.gfz-potsdam.de/geomultisens/spechomo/blob/master/LICENSE
.. image:: https://img.shields.io/pypi/pyversions/spechomo.svg
        :target: https://img.shields.io/pypi/pyversions/spechomo.svg

See also the latest coverage_ report and the nosetests_ HTML report.


Feature overview
----------------

SpecHomo is a **python package for spectral homogenization of multispectral satellite data**, i.e., for the transformation
of the spectral information of one sensor into the spectral domain of another one. This simplifies workflows, increases
the reliability of subsequently derived multi-sensor products and may also enable the generation of new products that
are not possible with the initial spectral definition.

SpecHomo offers **different machine learning techniques** for the prediction of the target sensor spectral information. So
far, multivariate linear regression, multivariate quadratic regression and random forest regression are implemented. To
allow easy comparisons to the most simple homogenization approach, we also implemented linear spectral interpolation.

In contrast to previous spectral homogenization techniques, SpecHomo not only allows to apply a global (band-wise)
transformation with the same prediction coefficients for all gray values of a spectral band. It also **distinguishes**
**between individual spectral characteristics of different land-cover types** by using specifically trained prediction
coefficients for various spectral clusters. This increases the accuracy of the predicted spectral information.
Apart from that, SpecHomo can not only be used to homogenize already similar spectral definitions - it also **allows to**
**predict unilaterally missing bands**.

**Prediction accuracies and effects to subsequent products** such as spectral indices or classifications have been
evaluated in the above mentioned paper at the example of Sentinel-2 spectral information predicted from Landsat-8.
Algorithm details may also be found there.

Satellite data (surface reflectance) acquired by **following sensors may be used** as source or target sensor:

* Landsat-5 TM
* Landsat-7 ETM+
* Landsat-8 OLI
* Sentinel-2A MSI
* Sentinel-2B MSI
* RapidEye-5 MSI
* SPOT-4
* SPOT-5

SpecHomo features **classifiers for homogenization** that we trained in the context of the GeoMultiSens project (see the
credits section) and for our evaluations related with the above mentioned paper. The initial spectral information for
classifier training has been derived from hyperspectral airborne data, spectrally convolved to different sensors. You
may also train your own homogenization classifiers specifically optimized to your area of interest. SpecHomo provides
the needed functionality for that.

For further details on how to use SpecHomo check out the
`documentation <http://geomultisens.gitext.gfz-potsdam.de/spechomo/doc/>`__!

Credits
-------

The spechomo package was developed within the context of the GeoMultiSens project funded
by the German Federal Ministry of Education and Research (project grant code: 01 IS 14 010 A-C).

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _coverage: http://geomultisens.gitext.gfz-potsdam.de/spechomo/coverage/
.. _nosetests: http://geomultisens.gitext.gfz-potsdam.de/spechomo/nosetests_reports/nosetests.html
