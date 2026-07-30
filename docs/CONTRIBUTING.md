# Contributing

Clone the repo and create your virtual environment from project root. This project requires Python 3.10+.

```bash
uv venv
source .venv/Scripts/activate || source .venv/bin/activate
uv sync --all-extras
pre-commit install
```

## Adding GLMs

To add a new GLM class, inherit from the `GLM` parent class (see `PoissonReg` and `LogReg` for examples). The `GLM` parent class provides a framework for implementing new models and should be used whenever possible. Planned but currently unimplemented GLMs include Negative Binomial, Gamma, and Tweedie.

The `GLM` parent class defines several empty instance methods that are intended to be overridden by child classes:

1. `self._objective_func`: The objective function to be minimized.
2. `self._link_func`: The GLM's link function (e.g., the logit link for Logistic Regression).
3. `self._grad_func`: The first derivative of the loss function, used by some optimization algorithms.
4. `self._hess_func`: The second derivative of the loss function, used by some optimization algorithms.

The implementations in the existing child classes are intentionally written for clarity to help readers and developers understand the underlying mathematics. This design also makes implementing new GLMs pretty straightforward.
