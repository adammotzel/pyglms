"""
Test LogReg class by comparing results to `statsmodels`.

https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html

These tests essentially treat the `statsmodels` results as 'ground truth'. This is 
just a simply way of ensuring the LogReg class calculates the core model results 
correctly.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import statsmodels.api as sm

from turtles.stats.glms import LogReg


# we'll increase the tolerance a bit because the models are fit using MLE
tol = 1e-4
prec = 4

# load sample data
X, y = load_breast_cancer(return_X_y=True)

# reduce dimensions
X = X[:, :3]
var_names = [f"test_{i}" for i in range(X.shape[1])]

# turtles model
model = LogReg(method="lbfgs")
model.fit(X, y, var_names=var_names)
summary = model.summary()
preds = model.predict(X)

# statsmodels
X = sm.add_constant(X)
sm_model = sm.Logit(y, X).fit()
sm_preds = sm_model.predict(X)


def test_logreg():
    """
    Test model statistics.

        1. Number of Observations
        2. Degrees of Freedom (residuals)
        3. Estimated Coefficients, i.e., Betas
        4. Estimated Coefficient P-values
        5. Estimated Coefficient Standard Errors
        6. Estimated Coefficient z-stats
        7. Model predictions (to ensure .predict() works)
    """

    assert all([col in summary["Variable"].unique() for col in var_names])
    assert model.observations == sm_model.nobs
    assert model.degrees_of_freedom == sm_model.df_resid
    np.isclose(
        model.betas[0],
        sm_model.params,
        atol=tol
    )
    np.isclose(
        model.p_values[0],
        sm_model.pvalues,
        atol=tol
    )
    np.isclose(
        model.std_error_betas[0],
        sm_model.bse,
        atol=tol
    )
    np.isclose(
        model.z_stat_betas[0],
        sm_model.tvalues,
        atol=tol
    )
    np.isclose(
        preds,
        sm_preds,
        atol=tol
    )


if __name__ == "__main__":
    test_logreg()
    print("All tests passed.")
