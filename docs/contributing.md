# Contributing Guidelines

Thank you for your interest in contributing to `kinetics-pose-har`! Contributions of all kinds are welcome — bug reports, documentation improvements, new features, and additional baseline models.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/kinetics-pose-har.git
   cd kinetics-pose-har
   ```
3. **Install** the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Code Style

This project uses [black](https://github.com/psf/black) for formatting and [flake8](https://flake8.pycqa.org/) for linting.

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/ --max-line-length 100
```

All pull requests must pass both checks.

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

Please ensure **all existing tests pass** and add new tests for any new functionality you introduce.

---

## Pull Request Process

1. Ensure your branch is up to date with `main`
2. Run `black`, `flake8`, and `pytest` — all must pass
3. Update `docs/` if you add or change any public API
4. Open a pull request against `main` with a clear description of the change

---

## Reporting Issues

Use [GitHub Issues](https://github.com/NajibaTagougui/kinetics-pose-har/issues) and include:

- Python version and OS
- Full error traceback
- Minimal reproducible example

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please be respectful and constructive in all interactions.

---

## Contact

For questions beyond GitHub Issues, contact the corresponding author:  
**Najiba Tagougui** — najiba.tagougui@isims.usf.tn
