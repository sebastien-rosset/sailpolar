# SailPolar

ML-based polar analysis tool for sailboats. This tool analyzes NMEA data to create accurate polar diagrams, taking into account:

- Instrument calibration and reliability
- Current effects
- Wave conditions
- Real-world performance factors

## Features

- Advanced polar diagram generation
- Current analysis and compensation
- Instrument reliability assessment
- Integration with OpenCPN
- ML-based data analysis

## Installation

```bash
pip install sailpolar
```

## Quick Start

```python
from sailpolar import PolarAnalyzer

# Initialize analyzer
analyzer = PolarAnalyzer()

# Load and analyze NMEA data
results = analyzer.analyze_file('path/to/nmea/log')

# Generate polar diagram
polar = results.generate_polar()

# Export to OpenCPN format
polar.export_opencpn('polar.csv')
```

## Development

Setup development environment:

```bash
git clone https://github.com/serosset/sailpolar.git
cd sailpolar
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

