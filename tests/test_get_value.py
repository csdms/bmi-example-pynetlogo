"""Test the BMI get_value functions."""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from heat import BmiHeatDiffusion

CONFIG_FILE = "config.yaml"
GRID_ID = 0
VAR_NAME = "plate_surface__temperature"


def test_get_value(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    dest0 = np.empty(model.get_grid_size(GRID_ID), dtype=float)
    dest1 = np.empty(model.get_grid_size(GRID_ID), dtype=float)

    model.get_value(VAR_NAME, dest0)
    model.get_value(VAR_NAME, dest1)

    assert dest0 is not dest1
    assert_array_almost_equal(dest0, dest1)

    model.finalize()


def test_get_value_ptr():
    model = BmiHeatDiffusion()
    with pytest.raises(NotImplementedError):
        model.get_value_ptr(VAR_NAME)


def test_get_value_at_indices():
    model = BmiHeatDiffusion()
    dest = np.empty(3, dtype=float)
    with pytest.raises(NotImplementedError):
        model.get_value_at_indices(VAR_NAME, dest, [0, 2, 4])


def test_value_size(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    dest = np.empty(model.get_grid_size(GRID_ID), dtype=float)
    model.get_value(VAR_NAME, dest)
    assert model.get_grid_size(0) == dest.size

    model.finalize()


def test_value_nbytes(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    dest = np.empty(model.get_grid_size(GRID_ID), dtype=float)
    model.get_value(VAR_NAME, dest)
    assert model.get_var_nbytes(VAR_NAME) == dest.nbytes

    model.finalize()
