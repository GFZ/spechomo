=======
History
=======

0.9.0 (2020-11-02)
------------------

* Replaced deprecated 'source activate' by 'conda activate.'
* Updated installation instructions.
* Revised requirements.
* Added doc, test, lint and dev requirements to optional requirements in setup.py.
* Updated LR and QR classifiers.
* Added sklearn import to avoid static TLS ImportError.
* Improved code style of SpectralHomogenizer.interpolate_cube() and SpectralHomogenizer.predict().
* Bugfix for also predicting spectral information for pixels that contain nodata in any band
  (causes faulty predictions).
* Bugfix for only choosing 25 spectra in classifier creation in case the maximum angle threshold is automatically
  set to 0 because there are many well matching spectra.
* Added minimal version of geoarray.


0.8.2 (2020-10-12)
------------------

* Use SPDX license identifier and set all files to GLP3+ to be consistent with license headers in the source files.


0.8.1 (2020-10-08)
------------------

* Added latest QR classifiers.


0.8.0 (2020-10-07)
------------------

* SpecHomo is now on conda-forge! Updated the installation instructions accordingly.


0.7.0 (2020-10-01)
------------------

* Re-trained LR classifiers.
* Updated classifiers within test data.
* Classifiers are no longer stored in the repository (resources directory) but are automatically downloaded on demand
  at the first run (added corresponding code).
* Fixed TemporaryDirectory bug in Test_Utils.test_export_classifiers_as_JSON().
* Re-enabled CI job 'deploy_pypi'.


0.6.10 (2020-09-25)
-------------------

* Fixed an AssertionError within ClusterClassifier_Generator.create_classifiers() caused by nodata pixels in the target
  sensor reference cube that were not dropped before creating the classifier.


0.6.9 (2020-09-25)
------------------

* Moved matplotlib imports function/class level to avoid static TLS ImportError.


0.6.8 (2020-09-25)
------------------

* Moved scipy imports function/class level to avoid static TLS ImportError.
* environment_spechomo.yml now uses Python 3.7+.
* scikit-learn is now pinned to 0.23.2+ due to classifier recreation.


0.6.7 (2020-09-24)
------------------

* Fixed a DeprecationWarning in case of scikit-learn>=0.23.
* Dumped regressors now use the second highest dill protocol in order to have some downwards compatibility.


0.6.6 (2020-09-24)
------------------

* Moved imports of scikit-learn to function/class level to avoid static TLS ImportError.


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
