Usage Guide
===========

Basic Usage
----------

Loading and Analyzing Data
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sailpolar import PolarAnalyzer

   analyzer = PolarAnalyzer()
   results = analyzer.analyze_file('path/to/nmea/log')

Current Analysis
--------------

The current analyzer can detect and compensate for tidal and ocean currents:

.. code-block:: python

   from sailpolar.analysis import CurrentAnalyzer

   current_analyzer = CurrentAnalyzer()
   current = current_analyzer.estimate_current(data)
   compensated_data = current_analyzer.compensate_for_current(data, current)

Instrument Analysis
-----------------

Monitor and analyze instrument health:

.. code-block:: python

   from sailpolar.analysis import InstrumentAnalyzer

   instrument_analyzer = InstrumentAnalyzer()
   health = instrument_analyzer.analyze_all_instruments(data)

Configuration
------------

Common configuration options and their effects...

Advanced Features
---------------

Description of advanced features and how to use them...

Best Practices
------------

Recommendations for getting the most accurate results...

Troubleshooting
-------------

Common issues and their solutions...