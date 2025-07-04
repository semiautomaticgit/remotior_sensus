Installation
===============

This section describes Remotior Sensus installation along with
the required dependencies.


Dependencies
____________

Remotior Sensus requires `GDAL`, `NumPy` and `SciPy` for most functionalities.
Optionally, `scikit-learn` and `PyTorch` are required for machine learning.
Python >= 3.10 is recommended.

.. _Installation with Conda:

Installation with Conda
_______________________

Before installing Remotior Sensus please install the dependencies using
a `Conda` environment (if you don't know `Conda` please read
https://conda-forge.org/docs).
For instance, you can use
`Miniforge <https://github.com/conda-forge/miniforge>`_
to create a `Conda` environment.

.. code-block:: console

    $ conda create -c conda-forge --name environment python=3.10
    Proceed ([y]/n)? y
    $ conda activate environment

Install Remotior Sensus using `Conda` (the fundamental dependencies are also installed):

.. code-block:: console

    $ conda install -c conda-forge remotior-sensus

For machine learning functionalities run:

.. code-block:: console

    $ conda install -c conda-forge remotior-sensus scikit-learn pytorch


Installation in Linux
_______________________

The suggested way to install Remotior Sensus is using `Conda` (see
`Installation with Conda`_).

Depending on the system, one could install the required dependencies as:

.. code-block:: console

    $ sudo apt-get install python3-numpy python3-scipy gdal-bin

For Remotior Sensus package installation use `pip`:

.. code-block:: console

    $ pip install -U remotior-sensus

Installation in OS X
____________________

The suggested way to install Remotior Sensus is using `Conda` (see
`Installation with Conda`_).


Installation in Windows
_______________________

The suggested way to install Remotior Sensus is using `Conda` (see
`Installation with Conda`_).


Package installation
____________________

Given that dependencies are installed, for Remotior Sensus package
installation use `pip`:

.. code-block:: console

    $ pip install -U remotior-sensus
