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
        <dd>Holds the model and a BMI for the model</dd>
    <dt>examples</dt>
        <dd>Jupyter Notebooks that demonstrate how to run the model standalone and through its BMI</dd>
</dl>

## Build/Install

## Use

## Acknowledgments

The model of temperature diffusion used in this example:

> Wilensky, U. (1998). NetLogo Heat Diffusion model. http://ccl.northwestern.edu/netlogo/models/HeatDiffusion. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

The NetLogo software:

> Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

CSDMS is supported with funding from the National Science Foundation.
