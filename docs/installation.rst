Installation
============

Requirements
------------

* Python ≥ 3.9
* NumPy ≥ 1.22
* SciPy

From PyPI
---------

.. code-block:: bash

   pip install simGL

From conda-forge
----------------

.. code-block:: bash

   conda install -c conda-forge simGL

From source (development)
--------------------------

.. code-block:: bash

   git clone https://github.com/RacimoLab/simGL.git
   cd simGL
   pip install -e .

The ``-e`` flag installs the package in editable mode, so any changes to the
source files are reflected immediately without reinstalling. To install without
editable mode (e.g. to test an installed release), use ``pip install .``.

Running the tests
-----------------

The test suite uses `msprime <https://tskit.dev/msprime/>`_ to generate
coalescent simulations. Install it alongside the test dependencies:

.. code-block:: bash

   pip install msprime pytest pytest-cov

Then run from the repository root:

.. code-block:: bash

   python -m pytest tests/
