import os
import shutil
import pytest
import logging
import numpy as np
from tests.download_resources import download_resources


@pytest.fixture(scope="session")
def test_dir():
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    # Always start fresh
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    download_resources(test_dir=test_dir)
    yield test_dir
    logging.info(f"Removing the temporary directory for tests.")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def dice() -> float:
    def _dice(pred: np.ndarray, gt: np.ndarray):
        intersection = np.logical_and(pred, gt).sum()
        dice = 2. * intersection / (pred.sum() + gt.sum())
        return dice
    return _dice