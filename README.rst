=====================================================
Knowledge Graph-Driven Image Data Conversion Code Generation
=====================================================

Introduction
------------

#todo - Provide a brief description of the tool, its purpose, key features, and any unique benefits or innovative aspects it offers.


Installation
------------

Install the tool directly using pip:

.. code-block:: bash

    pip install imgdataconvertcodegen

Or for a more manual installation:

.. code-block:: bash

    git clone git@github.com:c3di/ImgDataConvertCodeGen.git
    cd imgdataconvertcodegen
    python setup.py install

Usage
-----

.. code-block:: bash

   imgdataconvertcodegen  --help

Example:

.. code-block:: bash

    #todo imgdataconvertcodegen example_command -option1 -option2


Develop
-------

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository

.. code-block:: bash

    git clone git@github.com:c3di/ImgDataConvertCodeGen.git

Use the ``requirements_dev.txt`` file to install all necessary development dependencies.

.. code-block:: bash

    pip install -r requirements_dev.txt

Tests
~~~~~

There are two ways to run the unit tests for this project.

1. Using ``tox`` from the command line:

   .. code-block:: bash

       tox

   ``tox`` will download the dependencies from ``requirements_tests.txt``, build the package, install it in a virtual environment and run the tests using ``pytest``. For detailed configuration options, please go to `tox documentation <https://tox.wiki/en/stable/>`__.



2. Using ``pytest`` within the testing framework of an IDE like PyCharm:

   * Install all dependencies for tests:

     .. code-block:: bash

        pip install -r requirements_tests.txt

   * To run tests in the IDE, the project must be installed in editable mode.

     .. code-block:: bash

         pip install -e .

   * Set ``Dev`` environment variable to ``True`` in the IDE's run configuration.

     .. code-block:: none

         Dev=True

   * Run the tests through test runner interface of IDE.

Build
~~~~~
#todo

publish
~~~~~

#todo
License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

