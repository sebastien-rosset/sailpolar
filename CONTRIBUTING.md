# Contributing to SailPolar

First off, thank you for considering contributing to SailPolar! It's people like you that make SailPolar such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check [existing issues](https://github.com/serosset/sailpolar/issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include NMEA data samples if relevant (please anonymize sensitive data)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed feature
* Any possible alternatives you've considered
* Whether you'd be willing to help implement the enhancement

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python coding conventions
* Include tests for new features
* Document new code based on the Documentation section below

### Development Process

1. Fork the repo
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Style Guidelines

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Style Guide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use [Black](https://github.com/psf/black) for code formatting
* Use [isort](https://pycqa.github.io/isort/) for import sorting
* Use [MyPy](http://mypy-lang.org/) for type checking

### Documentation Style

* Use docstrings for all public modules, functions, classes, and methods
* Follow [Google style](https://google.github.io/styleguide/pyguide.html) for docstrings
* Keep documentation up to date with code changes
* Add examples for complex functionality

## Testing

* Write test cases for all new functionality
* Ensure all tests pass before submitting PR
* Include both unit tests and integration tests where appropriate

## Development setup

1. Clone your fork
2. Install development dependencies:
```bash
pip install -e ".[dev]"
```
3. Set up pre-commit hooks:
```bash
pre-commit install
```
4. Create a branch for local development:
```bash
git checkout -b name-of-your-feature
```

## Documentation

Documentation is written in Markdown for general project docs and reStructuredText for API documentation. Please make sure to update the documentation when changing code.

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

## Questions?

Feel free to open an issue with the question tag or contact the maintainers directly.