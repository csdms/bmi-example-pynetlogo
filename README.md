[![Basic Model Interface](https://img.shields.io/badge/CSDMS-Basic%20Model%20Interface-green.svg)](https://bmi.readthedocs.io/)

# bmi-example-pynetlogo

An example of using the
[Python bindings](https://github.com/csdms/bmi-python)
for the CSDMS
[Basic Model Interface](https://bmi.readthedocs.io) (BMI)
to wrap a model written in [NetLogo](https://ccl.northwestern.edu/netlogo/).

## Overview

This is an example of implementing a BMI for a simple model of temperature diffusion
on a uniform rectangular plate
with Dirichlet boundary conditions.
The model, [HeatDiffusion](https://ccl.northwestern.edu/netlogo/models/HeatDiffusion),
is written in NetLogo,
and is a part of the standard NetLogo distribution.

This repository is organized with the following directories:

<dl>
    <dt>heat</dt>
        <dd>Source for the model and a BMI implementation for the model</dd>
    <dt>examples</dt>
        <dd>Python scripts and Jupyter Notebooks that demonstrate how to run the model standalone and through its BMI</dd>
    <dt>tests</dt>
        <dd>Tests that cover the BMI of the model</dd>
</dl>

## Build/Install

This example can be built and installed on Linux, macOS, and Windows.

**Prerequisites:**

* NetLogo. Instructions for downloading and installing NetLogo can be found [here](https://ccl.northwestern.edu/netlogo/download.shtml). NetLogo 6.1.1 was used to build and test this example.
* The Python BMI bindings. Follow the [build and install directions](https://github.com/csdms/bmi-python#install) given in that repository. You can choose to install them from source, or through `pip` or `conda`.
* PyNetLogo and Jpype. These are the key software packages that allow communication between NetLogo and Python.

We recommend setting up a virtual environment--e.g., through `venv` or `conda`--to install the packages required for this example.
For `conda`,
we've included an environment file.
```
conda env create --file environment.yml
```
This create the *logo* environment,
which you can activate to have access to all the dependent packages needed to build and use this example.

Install the example with `pip`.
```
pip install -e .
```

## Use

## Acknowledgments

The model of temperature diffusion used in this example:

> Wilensky, U. (1998). NetLogo Heat Diffusion model. http://ccl.northwestern.edu/netlogo/models/HeatDiffusion. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

The NetLogo software:

> Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

CSDMS is supported with funding from the National Science Foundation.
