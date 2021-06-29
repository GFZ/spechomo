.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	## don't call coverage erase here because make install calls make clean which calls make clean-test
	## -> since make install should run without the test requirements we can't use coverage erase here
	rm -fr .tox/
	rm -f .coverage
	rm -fr .coverage.*
	rm -fr htmlcov/
	rm -fr nosetests.html
	rm -fr nosetests.xml

lint: ## check style with flake8
	flake8 --max-line-length=120 spechomo tests > ./tests/linting/flake8.log || \
		(cat ./tests/linting/flake8.log && exit 1)
	pycodestyle spechomo --exclude="*.ipynb,*.ipynb*" --max-line-length=120 > ./tests/linting/pycodestyle.log || \
		(cat ./tests/linting/pycodestyle.log && exit 1)
	-pydocstyle spechomo > ./tests/linting/pydocstyle.log || \
		(cat ./tests/linting/pydocstyle.log && exit 1)

urlcheck: ## check for dead URLs
	urlchecker check . --file-types .py,.rst,.md,.json

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage erase
	coverage run --source spechomo --source bin setup.py test
	coverage combine 	# must be called in order to make coverage work in multiprocessing
	coverage report -m
	coverage html
	# $(BROWSER) htmlcov/index.html

nosetests: clean-test ## Runs nosetests with coverage, xUnit and nose-html-output
	## - puts the coverage results in the folder 'htmlcov'
	## - generates 'nosetests.html' (--with-html)
	## - generates 'nosetests.xml' (--with-xunit) which is currently not visualizable by GitLab
	nosetests -vv --with-coverage --cover-package=spechomo --cover-erase --cover-html --cover-html-dir=htmlcov \
		--with-html --with-xunit --rednose --force-color

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/spechomo.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ spechomo
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	#$(MAKE) -C docs latex
	#$(MAKE) -C docs latexpdf
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install -r requirements.txt
	python setup.py install

gitlab_CI_docker:  ## Build a docker image for CI use within gitlab
	cd ./tests/gitlab_CI_docker/; bash ./build_spechomo_testsuite_image.sh
