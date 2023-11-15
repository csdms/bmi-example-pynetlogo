"""Test BMI model grid information functions."""
import numpy as np
from numpy.testing import assert_array_equal

from heat import BmiHeatDiffusion

CONFIG_FILE = "config.yaml"
GRID_ID = 0


def test_grid_functions(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    ndims = model.get_grid_rank(GRID_ID)
    assert ndims == 2

    gtype = model.get_grid_type(GRID_ID)
    assert gtype == "uniform_rectilinear"

    gsize = model.get_grid_size(GRID_ID)
    assert gsize == 2601

    gshape = np.empty(ndims, dtype=np.int32)
    model.get_grid_shape(GRID_ID, gshape)
    assert_array_equal(gshape, (51, 51))

    gspacing = np.empty(ndims, dtype=np.int32)
    model.get_grid_spacing(GRID_ID, gspacing)
    assert_array_equal(gspacing, (1.0, 1.0))

    gorigin = np.empty(ndims, dtype=np.int32)
    model.get_grid_origin(GRID_ID, gorigin)
    assert_array_equal(gorigin, (-25.0, -25.0))

    model.finalize()
