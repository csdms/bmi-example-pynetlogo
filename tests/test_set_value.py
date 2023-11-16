"""Test the BMI get_value functions."""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from heat import BmiHeatDiffusion

CONFIG_FILE = "config.yaml"
GRID_ID = 0
VAR_NAME = "plate_surface__temperature"


def test_set_value(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    z0 = np.empty(model.get_grid_size(GRID_ID), dtype=float)
    model.get_value(VAR_NAME, z0)
    z1 = np.zeros_like(z0) - 1

    model.set_value(VAR_NAME, z1)

    new_z = np.empty_like(z0)
    model.get_value(VAR_NAME, new_z)

    assert new_z is not z0
    assert new_z is not z1
    assert_array_almost_equal(new_z, z1)

    model.finalize()


def test_set_value_at_indices():
    model = BmiHeatDiffusion()
    with pytest.raises(NotImplementedError):
        model.set_value_at_indices(VAR_NAME, [0, 2, 4], [-1, -1, -1])
