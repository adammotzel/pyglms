"""
Test MLR class by comparing results to `statsmodels`.

https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html

Many of the tests essentially treat the `statsmodels` results as 'ground truth'. 
This is just a simple way of ensuring the MLR class calculates the core model results 
correctly.
"""

import pytest
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from scipy.stats import t

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

# calculate critical t-value
# SM OLS models don't offer this property
sm_critical_t = t.ppf(1 - 0.05/2, sm_model.df_resid)


def test_mlr():
    """
    Test model statistics (compare with `statsmodels`):

        1. Number of Observations
        2. Number of Dimensions
        3. Degrees of Freedom (residuals)
        4. Intercept
        5. Estimated Coefficients, i.e., Betas
        6. Residuals
        7. Variance
        8. RSS
        9. RMSE
        10. Standardized Residuals
        11. Covariance Matrix
        12. Critical t-value
        13. Confidence Intervals
        14. SST
        15. Mean Squared Error (residuals)
        16. Estimated Coefficient P-values
        17. Model R-squared
        18. Model Adjusted R-dquared
        19. Estimated Coefficient t-stats
        20. Estimated Coefficient Standard Errors
        21. Model predictions (to ensure .predict() works)
    
    Other tests:

        22. _is_fit()
        23. Variable names

    """

    # ----- Compare to statsmodels -----

    assert all([col in summary["Variable"].unique() for col in var_names])
    assert model.observations == sm_model.nobs
    assert model.dimensions == X.shape[1] - 1
    assert model.degrees_of_freedom == sm_model.df_resid

    assert np.isclose(
        model.intercept,
        sm_model.params[0],
        atol=tol
    ).all()
    assert np.isclose(
        model.betas[0],
        sm_model.params,
        atol=tol
    ).all()
    assert np.isclose(
        model.residuals.flatten(),
        sm_model.resid,
        atol=tol
    ).all()
    assert np.isclose(
        model.variance,
        sm_model.scale,
        atol=tol
    ).all()
    assert np.isclose(
        model.rss,
        sm_model.ssr,
        atol=tol
    ).all()
    assert np.isclose(
        model.rmse,
        np.sqrt(sm_model.mse_resid),
        atol=tol
    ).all()
    assert np.isclose(
        model.std_residuals.flatten(),
        sm_model.resid_pearson,
        atol=tol
    ).all()
    assert np.isclose(
        model.covariance,
        sm_model.cov_params(),
        atol=tol
    ).all()
    assert np.isclose(
        model.critical_t,
        sm_critical_t,
        atol=tol
    ).all()
    assert np.isclose(
        model.confidence_interval[0][0],
        sm_model.conf_int()[:,0],
        atol=tol
    ).all()
    assert np.isclose(
        model.confidence_interval[1][0],
        sm_model.conf_int()[:,1],
        atol=tol
    ).all()
    assert np.isclose(
        model.sst,
        sm_model.centered_tss,
        atol=tol
    ).all()
    assert np.isclose(
        model.mse, 
        sm_model.mse_resid,
        atol=tol
    )
    assert np.isclose(
        model.p_values[0],
        sm_model.pvalues,
        atol=tol
    ).all()
    assert np.isclose(
        model.r2, 
        sm_model.rsquared,
        atol=tol
    ).all()
    assert np.isclose(
        model.r2_adj, 
        sm_model.rsquared_adj,
        atol=tol
    ).all()
    assert np.isclose(
        model.t_stat_betas[0],
        sm_model.tvalues,
        atol=tol
    ).all()
    assert np.isclose(
        model.std_error_betas[0],
        sm_model.bse,
        atol=tol
    ).all()
    assert np.isclose(
        preds.flatten(),
        sm_preds,
        atol=tol
    ).all()

    # ----- General testing -----

    # check model fit
    dummy_model = MLR()
    with pytest.raises(Warning):
        dummy_model._is_fit()

    # incorrect number of variable names
    with pytest.raises(ValueError):
        dummy_model.fit(X=X, y=y, var_names=["test"])
