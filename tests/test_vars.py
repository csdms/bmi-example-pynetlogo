"""Test BMI model variable information functions."""

from heat import BmiHeatDiffusion

CONFIG_FILE = "config.yaml"
VAR_NAME = "plate_surface__temperature"


def test_var_functions(shared_datadir):
    model = BmiHeatDiffusion()
    model.initialize(shared_datadir / CONFIG_FILE)

    id = model.get_var_grid(VAR_NAME)
    assert id == 0

    dtype = model.get_var_type(id)
    assert dtype == "float64"

    units = model.get_var_units(id)
    assert units == "C"

    isize = model.get_var_itemsize(id)
    assert isize == 8

    nbytes = model.get_var_nbytes(id)
    assert nbytes == 20808

    loc = model.get_var_location(id)
    assert loc == "face"

    model.finalize()
