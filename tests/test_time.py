"""Test BMI model time functions."""
import numpy as np
from numpy.testing import assert_almost_equal

from heat import BmiHeatDiffusion


def test_start_time():
    model = BmiHeatDiffusion()

    assert_almost_equal(model.get_start_time(), 0.0)


def test_end_time():
    model = BmiHeatDiffusion()

    assert_almost_equal(model.get_end_time(), np.finfo("d").max)


def test_current_time():
    model = BmiHeatDiffusion()

    assert_almost_equal(model.get_current_time(), 0.0)


def test_time_step():
    model = BmiHeatDiffusion()

    assert_almost_equal(model.get_time_step(), 0.1)


def test_time_units():
    model = BmiHeatDiffusion()

    units = model.get_time_units()
    assert units == "s"
