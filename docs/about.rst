=====
About
=====

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
**predict unilaterally missing bands** such as the red edge bands that are not present in Landsat-8 data.

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
`documentation <https://geomultisens.gitext-pages.gfz-potsdam.de/spechomo/doc/>`__!
