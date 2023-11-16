"""Test BMI model control functions."""
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from heat import BmiHeatDiffusion

CONFIG_FILE = "config.yaml"
GRID_SHAPE = (51, 51)


# Gang tests because finalize must be paired with initialize, else pynetlogo hangs.
def test_irf(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    ndim = model.get_grid_rank(0)
    shape = np.empty(ndim, dtype=np.int32)

    assert_array_equal(model.get_grid_shape(0, shape), GRID_SHAPE)

    for inc in range(10):
        model.update()
        assert_almost_equal(model.get_current_time(), (inc + 1) * model.get_time_step())

    model.finalize()
