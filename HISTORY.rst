=======
History
=======

0.6.7 (2020-09-24)
------------------

* Fixed a DeprecationWarning in case of scikit-learn>=0.23.
* Dumped regressors now use the second highest dill protocol in order to have some downwards compatibility.


0.6.6 (2020-09-24)
------------------

* Moved imports of scikit-learn to function/class level to avoid static TLS ImportError. Updated version info.


0.6.5 (2020-09-15)
------------------

* Replaced deprecated HTTP links.


0.6.4 (2020-04-09)
------------------

* Fixed test_spechomo_install CI job.


0.6.3 (2020-04-09)
------------------

* Fixed create_github_release CI job.


0.6.2 (2020-04-09)
------------------

* Releases in the GitHub-Mirror-Repository are now created automatically
  (added create_release_from_gitlab_ci.sh and create_github_release CI job).
* Added GitHub issue template.


0.6.1 (2020-04-07)
------------------

* Revised CITATION file and .zenodo.json.


0.6.0 (2020-04-04)
------------------

* Added functionality to export existing .dill classifiers to JSON format to make them also usable in different
  programming environments.
* The documentation now contains links to the published version of the research paper corresponding to SpecHomo.
* Changed Zenodo title and description.
* Fixed fallback algorithm in SpectralHomogenizer.predict_by_machine_learner() and added corresponding tests.
* SpectralHomogenizer.interpolate_cube() now returns a GeoArray instead of a numpy array.


0.5.0 (2020-02-20)
------------------

* Removed pyresample dependency (not needed anymore).
* Updated README.rst and setup.py.
* Pinned geopandas to below version 0.6.3 to fix an incompatibility with pyproj.
* Updated CI runner setup scripts and CI jobs.
* Updated LR and QR classifiers.


0.4.0 (2019-10-07)
------------------

* Added Sphinx documentation.
* Improved usability by adding functions to explore available spectral tansformations.


0.3.0 (2019-09-25)
------------------

* All tests are working properly now.
* Added license texts.
* Revised global classifiers.
* Added harmonization using weighted averaging.


0.2.0 (2019-07-22)
------------------

* A lot of algorithm improvements. Refer to the commits for details.


0.1.0 (2019-03-26)
------------------

* First version working separately from geomultisens.
