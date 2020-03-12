Requirements to your input data
*******************************

Compatible image formats
~~~~~~~~~~~~~~~~~~~~~~~~

SpecHomo uses the `geoarray <https://gitext.gfz-potsdam.de/danschef/geoarray>`__ library for reading and writing raster
data which is built on top of GDAL. So the input images can have any GDAL compatible image format.
You can find a list here:
http://www.gdal.org/formats_list.html


Compatible sensors
~~~~~~~~~~~~~~~~~~

Satellite data acquired by the following sensors may be used as source or target sensor:

* Landsat-5 TM
* Landsat-7 ETM+
* Landsat-8 OLI
* Sentinel-2A MSI
* Sentinel-2B MSI
* RapidEye-5 MSI
* SPOT-4
* SPOT-5


Surface reflectance
~~~~~~~~~~~~~~~~~~~

For homogenization using the classifiers included in the SpecHomo repository, the input images must contain surface
reflectance data. This is because these classifiers have been trained with surface reflectance data to avoid gray value
differences due to different acquisition/illumination conditions or atmospheric states. See
`Scheffler et al. 2020 <https://doi.org/10.1016/j.rse.2020.111723>`__ for more information.


Required image metadata
~~~~~~~~~~~~~~~~~~~~~~~


.. todo:


- no-data value

