"""Test BMI model information functions."""

from heat import BmiHeatDiffusion


def test_component_name():
    model = BmiHeatDiffusion()

    name = model.get_component_name()
    assert name == "The 2D Heat Equation"
    assert model.get_component_name() is name


def test_input_item_count():
    model = BmiHeatDiffusion()

    count = model.get_input_item_count()
    assert count == 0


def test_input_var_names():
    model = BmiHeatDiffusion()

    names = model.get_input_var_names()
    assert names == ()


def test_output_item_count():
    model = BmiHeatDiffusion()

    count = model.get_output_item_count()
    assert count == 1


def test_output_var_names():
    model = BmiHeatDiffusion()

    names = model.get_output_var_names()
    assert names == ("plate_surface__temperature",)
