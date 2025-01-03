Welcome to SailPolar's documentation!
================================

SailPolar is a ML-based polar analysis tool for sailboats that processes NMEA data to create accurate polar diagrams.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   configuration
   examples
   contributing
   changelog

Quick Start
----------

Installation
^^^^^^^^^^^

.. code-block:: bash

   pip install sailpolar

Basic Usage
^^^^^^^^^^

.. code-block:: python

   from sailpolar import PolarAnalyzer

   # Initialize analyzer
   analyzer = PolarAnalyzer()

   # Load and analyze NMEA data
   results = analyzer.analyze_file('path/to/nmea/log')

   # Generate polar diagram
   polar = results.generate_polar()

   # Export to OpenCPN format
   polar.export_opencpn('polar.csv')

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`