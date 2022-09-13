# temo

## Why ``temo``?

Fitting thermodynamic models is a challenging task, especially for mixtures with high non-ideality.  This process is usually not well documented in the literature, and the goal is to develop a set of tools that makes mixture model fitting something that is accessible to a wide range of users.  The focus of the development of the fitting tools has been for the development of mixture models for blends of refrigerants, but the tools are written in a very generic way.

Writing the fitter in Python allows for a high-level approach that is performant, parallelizable, and cross-platform. There is a low-level connection with the new teqp thermodynamic library, which allows great freedom in terms of what models are to be optimized.

The fitter is intended to fix mixture models in the corresponding states framework, which is sometimes referred to as the GERG approach, or a multi-fluid model. This approach is used in NIST REFPROP, CoolProp, and TREND (and a few others).  The other models in teqp (e.g. PC-SAFT) could also be optimized with the tools in this work with only minor modifications.

# Pre-requisites

In your working environment, you will need

- teqp
- scipy
- numpy
- matplotlib

A conda environment file is provided, but you may need a more updated version depending on what sort of data you are fitting.  With the default environment with conda, you can do:

```
conda env create -f .\environment.yml
```

to create the ``temo`` conda environment

# Installation

For now, as temo is not pushed to pypi, install with git:

```
pip install git+https://github.com/usnistgov/temo.git
```

Or, if you want to edit or make changes to temo, clone it and build in development mode:
```
git clone https://github.com/usnistgov/temo
cd temo
pip install --editable .
```

# Contact information
   - Ian Bell, MML, Div 647, ian.bell@nist.gov

## LICENSE

See [LICENSE.md](LICENSE.md) included in this repository