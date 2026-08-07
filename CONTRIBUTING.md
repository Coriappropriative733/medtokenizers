# Contributing to medtokenizers

Thank you for your interest in contributing to **medtokenizers**! This project
provides reusable tokenization modules for volumetric medical imagery, and we
welcome bug reports, feature requests, documentation improvements, and code
contributions from the community.

By contributing to this project, you agree that your contributions will be
licensed under the [MIT License](LICENSE), the same license that
covers the project. Please also read and abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

`medtokenizers` targets Python 3.10+ and PyTorch 2.0+. We recommend developing
inside a fresh virtual environment.

### Using pip

```bash
# Clone your fork and enter the repository
git clone https://github.com/<your-username>/medtokenizers.git
cd medtokenizers

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with the test extras
pip install -e ".[test]"
```

To install every optional dependency group (test, cloud, and training tooling)
at once:

```bash
pip install -e ".[all]"
```

### Using uv

If you prefer [uv](https://github.com/astral-sh/uv), the equivalent commands are:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[test]"
```

The project pins PyTorch to the CUDA 12.6 wheel index in `pyproject.toml` under
`[tool.uv.sources]`; uv will resolve `torch` from that index automatically.

---

## Running the Tests

The test suite uses [pytest](https://docs.pytest.org/) and lives under `tests/`.
Run the full suite with:

```bash
pytest
```

The pytest configuration enables `--strict-markers` and `--strict-config`, so
any unknown marker is treated as an error. Keep the following in mind when
running or writing tests:

- **Network tests.** Some tests download pretrained weights (for example, the
  LPIPS/VGG feature backbones). These are **skipped by default** so the suite
  stays hermetic and CI-friendly. To opt in, set the `MEDTOK_RUN_NETWORK_TESTS`
  environment variable:

  ```bash
  MEDTOK_RUN_NETWORK_TESTS=1 pytest
  ```

- **GPU tests.** Tests that require CUDA are skipped on CPU-only machines. Run
  them on a GPU-enabled host to exercise the full code path.

- **Slow tests.** End-to-end and full-training tests can take a long time. When
  iterating locally, you can deselect them by marker, for example:

  ```bash
  pytest -m "not slow"
  ```

Run a single test file or test while iterating:

```bash
pytest tests/test_quantizers.py
pytest "tests/test_quantizers.py::TestGradientFlow"
```

Coverage is available via `pytest-cov`:

```bash
pytest --cov=medtokenizers
```

---

## Code Style and Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, wired up
through [pre-commit](https://pre-commit.com/) so checks run automatically on every
commit. Install the hooks once after cloning:

```bash
pip install pre-commit
pre-commit install
```

You can run the hooks against all files at any time:

```bash
pre-commit run --all-files
```

Or invoke ruff directly:

```bash
ruff check .
ruff format .
```

Additional style conventions:

- 4-space indentation.
- [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Type hints throughout. Several modules use `jaxtyping` together with the
  `@jaxtyped_compile_safe(beartype)` decorator from `modules/utils.py`
  (see `modules/quant.py` for the canonical pattern); follow that style where it
  is appropriate.
- Use the standard library `logging` module for diagnostics rather than `print`:

  ```python
  import logging

  logger = logging.getLogger(__name__)
  logger.info("...")
  ```

---

## Issue and Pull Request Flow

### Reporting issues

- Search the [issue tracker](https://github.com/liamchalcroft/medtokenizers/issues)
  first to avoid duplicates.
- For bugs, include a minimal reproducible example, the full traceback, and your
  environment details (OS, Python version, PyTorch/CUDA version, package version).
- For feature requests, describe the use case and the behaviour you would like to
  see.
- Please do **not** file security vulnerabilities as public issues. See
  [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

### Submitting pull requests

1. Fork the repository and create a feature branch off `main`
   (`git checkout -b my-feature`).
2. Make your changes, following the code style above.
3. Add or update tests to cover your change.
4. Ensure the test suite passes (`pytest`) and the pre-commit hooks are clean
   (`pre-commit run --all-files`).
5. Update documentation and the `[Unreleased]` section of
   [CHANGELOG.md](CHANGELOG.md) as appropriate.
6. Push your branch and open a pull request against `main`, describing the
   motivation and summarizing the changes. Link any related issues.

A maintainer will review your pull request and may request changes. Once
approved and green, it will be merged. Thank you for helping improve
medtokenizers!

---

## License of Contributions

All contributions to this repository are made under the terms of the
[MIT License](LICENSE). By opening a pull request you certify
that you have the right to submit the work and that you license it to the project
and its users under those terms.
