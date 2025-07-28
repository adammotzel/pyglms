"""
Test MLR class by comparing results to `statsmodels`.

https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html

These tests essentially treat the `statsmodels` results as 'ground truth'. This is 
just a simply way of ensuring the MLR class calculates the core model results 
correctly.
"""

import numpy as np
from sklearn.datasets import load_diabetes
import statsmodels.api as sm

from turtles.stats.glms import MLR


tol = 1e-5
prec = 5

# sample data
X, y = load_diabetes(return_X_y=True)
y = y.reshape(y.shape[0], 1)
var_names = [f"test_{i}" for i in range(X.shape[1])]

# instantiate turtles model
model = MLR()

# fit model using all predictors
model.fit(X=X, y=y, var_names=var_names)
summary = model.summary()
preds = model.predict(X)

# statsmodels model
X = sm.add_constant(X)
sm_model = sm.OLS(y, X).fit()
sm_preds = sm_model.predict(X)


def test_mlr():
    """
    Test model statistics.

        1. Number of Observations
        2. Degrees of Freedom (residuals)
        3. Estimated Coefficients, i.e., Betas
        4. Mean Squared Error (residuals)
        5. Estimated Coefficient P-values
        6. Model R-squared
        7. Model Adjusted R-dquared
        8. Estimated Coefficient t-stats
        9. Estimated Coefficient Standard Errors
        10. Model predictions (to ensure .predict() works)
    """

    assert all([col in summary["Variable"].unique() for col in var_names])
    assert model.observations == sm_model.nobs
    assert model.degrees_of_freedom == sm_model.df_resid
    np.isclose(
        model.betas[0],
        sm_model.params,
        atol=tol
    )
    np.testing.assert_approx_equal(
        model.mse, 
        sm_model.mse_resid,
        prec
    )
    np.isclose(
        model.p_values[0],
        sm_model.pvalues,
        atol=tol
    )
    np.testing.assert_approx_equal(
        model.r2, 
        sm_model.rsquared,
        prec
    )
    np.testing.assert_approx_equal(
        model.r2_adj, 
        sm_model.rsquared_adj,
        prec
    )
    np.isclose(
        model.t_stat_betas[0],
        sm_model.tvalues,
        atol=tol
    )
    np.isclose(
        model.std_error_betas[0],
        sm_model.bse,
        atol=tol
    )
    np.isclose(
        preds,
        sm_preds,
        atol=tol
    )


if __name__ == "__main__":
    test_mlr()
    print("All tests passed.")
