"""
Test PoissonReg class by comparing results to `statsmodels`.

https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Poisson.html

These tests essentially treat the `statsmodels` results as 'ground truth'. This is 
just a simply way of ensuring the PoissonReg class calculates the core model results 
correctly.
"""

import statsmodels.api as sm
import numpy as np
from sklearn.datasets import fetch_openml

from turtles.stats.glms import PoissonReg
from turtles.preprocess import one_hot_encode


# we'll increase the tolerance a bit because the models are fit using MLE
tol = 1e-4
prec = 4

# get insurance claims data (cut the dataset for simplicity)
df = fetch_openml(data_id=41214, as_frame=True).frame
df = df.iloc[:5000]
df = df[["ClaimNb", "Exposure", "Area", "VehBrand"]].copy()
df_enc = one_hot_encode(df, ["Area", "VehBrand"])

# exog and endog
X = df_enc.drop(columns=["ClaimNb"])
y = df_enc["ClaimNb"]

# extract exposure
exposure = X["Exposure"]
X = X.drop(columns=["Exposure"])

# convert to numpy
X = X.to_numpy()
exposure = exposure.to_numpy().reshape(exposure.shape[0], 1)
y = y.to_numpy().reshape(y.shape[0], 1)

var_names = [f"test_{i}" for i in range(X.shape[1])]

# fit model using l-bfgs
model = PoissonReg(method="lbfgs")
model.fit(X, y, exposure, var_names=var_names)
summary = model.summary()
preds = model.predict(X)

# fit statsmodels
X = sm.add_constant(X)
y = y.flatten()
exposure = exposure.flatten()
sm_model = sm.GLM(
    y, 
    X, 
    exposure=exposure,
    family=sm.families.Poisson()
).fit()
sm_preds = sm_model.predict(X)


def test_poissonreg():
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
    test_poissonreg()
    print("All tests passed.")
