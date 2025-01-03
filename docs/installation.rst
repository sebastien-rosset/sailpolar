Installation
============

Prerequisites
------------

Before installing SailPolar, ensure you have Python 3.8 or later installed on your system.

Basic Installation
----------------

You can install SailPolar using pip:

.. code-block:: bash

   pip install sailpolar

Development Installation
----------------------

For development, you'll want to install the package with development dependencies:

.. code-block:: bash

   git clone https://github.com/serosset/sailpolar.git
   cd sailpolar
   pip install -e ".[dev]"

This will install additional dependencies needed for development, including:

- pytest for testing
- black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for code linting

Docker Installation
-----------------

We also provide a Docker image for easy deployment:

.. code-block:: bash

   docker pull serosset/sailpolar
   docker run -it serosset/sailpolar

Troubleshooting
--------------

Common installation issues and their solutions...

System-specific Notes
-------------------

Windows
^^^^^^^
Special considerations for Windows installations...

macOS
^^^^^
Special considerations for macOS installations...

Linux
^^^^^
Special considerations for Linux installations...