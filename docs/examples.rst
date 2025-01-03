Examples
========

Basic Examples
------------

Analyzing a Single NMEA Log
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sailpolar import PolarAnalyzer

   analyzer = PolarAnalyzer()
   results = analyzer.analyze_file('path/to/nmea/log')
   polar = results.generate_polar()
   polar.export_opencpn('polar.csv')

Current Analysis
--------------

Detecting and Compensating for Current
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sailpolar.analysis import CurrentAnalyzer

   analyzer = CurrentAnalyzer()
   current = analyzer.estimate_current(data)
   print(f"Detected current: {current.speed}kts at {current.direction}°")

Advanced Examples
---------------

Real-time Analysis
^^^^^^^^^^^^^^^^

Example of processing NMEA data in real-time...

Integration with OpenCPN
^^^^^^^^^^^^^^^^^^^^^

Example of integrating with OpenCPN...

Complete Application
^^^^^^^^^^^^^^^^^

A complete example showing all features together...