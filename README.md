![CI](https://github.com/adammotzel/pyglms/actions/workflows/ci.yaml/badge.svg)
![coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%20--%203.13-blue)
![License](https://img.shields.io/github/license/adammotzel/pyglms)



# PyGLMs

An implementation of various Generalized Linear Models (GLMs), written in Python.

I created this package as a refresher on GLMs and the underlying optimization techniques. It's intended as a learning tool and a reference for building and understanding these models from the ground up.


## Overview

The code is packaged as a Python library named `turtles` ([I like turtles](https://www.youtube.com/watch?v=CMNry4PE93Y)), making the code easy to integrate into your own projects.

The package is written using `numpy` for linear algebra operations, `scipy` for (some) optimization, `pandas` for displaying tabular results, and `matplotlib` for plots.

The following models have been implemented:

1. Multiple Linear Regression (`turtles.stats.glms.MLR` class)
2. Logistic Regression (`turtles.stats.glms.LogReg` class, uses `GLM` parent class)
3. Poisson Regression (`turtles.stats.glms.PoissonReg` class, uses `GLM` parent class)

The `GLM` parent class supports three optimization methods for parameter estimation: Momentum-based Gradient Descent for first-order optimization, Newton's Method for second-order optimization, and Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS). The user can specify the desired optimization `method` during class instantiation.

Momentum-based Gradient Descent and Newton's Method are implemented in Python as part of the `turtles` distribution. L-BFGS is implemented using `scipy.optimize`; it's a quasi-Newton method that approximates the Hessian (instead of fully computing it, like Newton's Method), so it's quite fast.

See `examples/{class name}_example.ipynb` for simple examples of using each model class and various supporting functions.


## Contributing

To run (and edit) this project locally, clone the repo and create your virtual environment from project root using your global (or local) Python version. This project requires Python 3.10+.

```bash
python -m venv venv
```

Activate the env (`source venv/Scripts/activate` for Windows OS, `source venv/bin/activate` for Linux) and install dependencies:

```bash
pip install -e .[dev]
```

Optionally, you can execute `scripts/env.sh` to create and activate a virtual environment using `uv`. The `uv` package manager must be installed for this to work.


### Adding GLMs

To add more GLM classes, use the `GLM` parent class for inheritence (see `PoissonReg` and `LogReg` as examples). The GLM parent class provides a solid framework for implementing new child classes and should be used whenever possible. Unimplemented GLMs include Negative Binomial, Gamma, and Tweedie.


## Testing

All tests are contained within `tests` directories for each module. You can simply execute the `pytest` command from project root to run all unit tests.

```bash
pytest
```

**Notes on Test Coverage:**
- Plotting functions from `turtles.plotting` are tested, but plotting methods in GLM classes 
(like `MLR`) are ignored. Those class methods are essentially just wrappers around `matplotlib` 
and `turtles.plotting` functions.
- `GLM` class methods that are meant to be implemented by child classes are ignored.
