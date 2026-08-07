"""Pytest configuration for medtokenizers tests."""

import os
import random
import sys
from pathlib import Path

import pytest

# Add repo root to path so scripts can be imported as a package
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# Network tests are opt-in and hermetic: rather than probing for connectivity
# at import time (which is slow, flaky, and couples the suite to external
# hosts), they only run when the user explicitly sets MEDTOK_RUN_NETWORK_TESTS.
# The symbol name is kept stable because other test modules import it.
requires_network = pytest.mark.skipif(
    not os.environ.get("MEDTOK_RUN_NETWORK_TESTS"),
    reason="network tests disabled; set MEDTOK_RUN_NETWORK_TESTS=1",
)


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed torch/numpy/random before every test for deterministic outputs.

    Autouse keeps the suite reproducible without each test having to manage
    its own seeding. Tests that need a specific seed may still re-seed locally.
    """
    seed = 0
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep, but stay defensive in CI
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    yield
