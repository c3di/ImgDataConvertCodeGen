=====================================================
imgdataconvertcodegen Development
=====================================================

Setup Dev Environment
--------------------------------------

**Clone the Repository**::

    git clone git@github.com:c3di/ImgDataConvertCodeGen.git

**Installing Dependencies**

Before installing dependencies, please ensure:

- For TensorFlow with CUDA support, verify system compatibility at `TensorFlow's installation guide <https://www.tensorflow.org/install/pip>`_.
- Use the correct path for ``path/to/requirements.txt`` in your project when executing the installation command.

With the Python virtual environment activated, install the required dependencies::

    pip install -r path/to/requirements.txt

**Running Tests**

Navigate to the project's working directory and install the project in editable mode::

    pip install -e .

To run the tests, use the following command::

    pytest

or run the tests through test runner interface of IDE like PyCharm or Visual Studio Code.

Build
--------------------------------------

To build the package, use the following command::

    tox -e build

Also you can remove old distribution files and temporary build artifacts (./build and ./dist) using the following command::

    tox -e clean

Publish
--------------------------------------
The version number of this project is automatically determined based on the latest git tag through ``setuptools_scm``."
To create a new version, create a new tag and push it to the repository::

    git tag -a v0.1.0 -m "Version 0.1.0"
    git push origin v0.1.0

To publish the package to a package index server, use the following command::

    tox -e publish

By default, it uses ``testpypi``. If you want to publish the package
to be publicly accessible in PyPI, use the ``-- --repository pypi`` option

