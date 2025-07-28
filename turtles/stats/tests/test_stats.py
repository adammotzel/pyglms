"""
Unit tests for functions from the stats module.
"""

import pytest
import numpy as np

from ...stats import (
    covariance_matrix,
    pca,
    pearson_corr,
    variance_inflation_factor,
    calculate_errors
)


# generate random data for testing
np.random.seed(11)
X = np.random.rand(10, 5)
prec = 8

# ----- covariance_matrix() -----

true_cov = np.array(
    [
        [ 0.08407138,  0.01703046, -0.00241016,  0.00151463,  0.00730384],
        [ 0.01703046,  0.14481672,  0.01960091, -0.05035906,  0.02466841],
        [-0.00241016,  0.01960091,  0.07601278,  0.04428605,  0.02239362],
        [ 0.00151463, -0.05035906,  0.04428605,  0.08308211,  0.01975682],
        [ 0.00730384,  0.02466841,  0.02239362,  0.01975682,  0.07193656]
    ]
)

true_xc = np.array(
    [
        [-0.27729601,  0.0278614 ,  0.27239877,  0.17476832,  0.30141389,      
        -0.37361255, -0.05588922,  0.4915367 , -0.39387927, -0.16730203],
       [-0.44870937, -0.4554038 , -0.35944854, -0.447701  ,  0.35009075,
         0.24454132,  0.37979438,  0.51848872, -0.10356897,  0.32191651],
       [-0.03045405, -0.00630097,  0.40023159, -0.37693531, -0.14904809,
         0.10587082,  0.2241766 , -0.15561853, -0.42364978,  0.41172774],
       [ 0.20806769,  0.42494041,  0.340288  , -0.20049893, -0.19806745,
        -0.46119256,  0.08519781, -0.27699156, -0.19749854,  0.27575514],
       [ 0.0035559 ,  0.43414739, -0.25156108, -0.25873539, -0.30498647,
         0.06314958,  0.13573612,  0.37978805, -0.34626511,  0.14517101]
    ]
)

true_pca = np.array(
    [
        [ 1.25533094, -0.03519149],
        [ 1.23540226,  0.98488766],
        [ 1.06981702,  0.79705956],
        [ 0.74358404, -1.46176967],
        [-0.94971483, -0.69670306],
        [-0.86189177, -0.42455926],
        [-0.72634048,  0.86364485],
        [-1.67673234,  0.20334144],
        [ 0.28137164, -1.66086366],
        [-0.37082647,  1.43015362]
    ]
)


def test_covariance_matrix():
    """Test covariance_matrix function."""

    C = covariance_matrix(X)
    np.testing.assert_array_equal(
        np.round(C, prec),
        true_cov
    )


def test_covariance_matrix_xc():
    """Test covariance matrix function when X_c is requested."""

    C, X_c = covariance_matrix(X, return_xc=True)
    np.testing.assert_array_equal(
        np.round(C, prec),
        true_cov
    )
    np.testing.assert_array_equal(
        np.round(X_c, prec),
        true_xc
    )


def test_covariance_invalid_case():
    """Test invalid case."""

    with pytest.raises(ValueError):
        covariance_matrix(X[:, 0])


# ----- pca() -----


def test_pca():
    """Test PCA function."""

    result = pca(X, 2)
    np.testing.assert_array_equal(
        np.round(result, prec),
        true_pca
    )
    result = pca(X, 1)
    np.testing.assert_array_equal(
        np.round(result, prec),
        np.array([true_pca[:,0]]).T
    )


def test_pca_invalid_case():
    """Test invalid cases."""

    with pytest.raises(ValueError):
        pca(X[:, 0], 1)

    with pytest.raises(ValueError):
        pca(X[:, :3], 5)


# ----- pearson_corr() -----


def test_pearson_corr_valid():
    """Test valid case."""

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    # should be 1 because the relationship is perfectly linear
    result = pearson_corr(x, y)

    assert np.isclose(result[0], 1.0)
    assert np.isclose(result[1], 0.0)


def test_pearson_corr_no_correlation():
    """Test valid case."""

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])

    #  should be -1, since there's a perfect inverse relationship
    result = pearson_corr(x, y)

    assert np.isclose(result[0], -1.0)
    assert np.isclose(result[1], 0.0)


def test_pearson_corr_invalid_case():
    """Test invalid case."""

    with pytest.raises(ValueError):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([[5, 4], [3, 2]])
        pearson_corr(x, y)


# ----- calculate_errors() -----


def test_calculate_errors_valid():
    """Valid input."""
    
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2.5, 5.5, 6.8, 9.2])
    
    # calculate manually for validation:
    errors = y_true - y_pred
    sse = np.sum(errors ** 2)
    mse = sse / len(y_true)
    rmse = np.sqrt(mse)

    result = calculate_errors(y_true, y_pred)

    assert np.isclose(result["Sum of Squared Errors"], sse)
    assert np.isclose(result["Mean Squared Error"], mse)
    assert np.isclose(result["Root Mean Squared Error"], rmse)


def test_calculate_errors_no_error():
    """Test case with no error."""

    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([3, 5, 7, 9])
    
    result = calculate_errors(y_true, y_pred)
    
    assert np.isclose(result["Sum of Squared Errors"], 0.0)
    assert np.isclose(result["Mean Squared Error"], 0.0)
    assert np.isclose(result["Root Mean Squared Error"], 0.0)


def test_calculate_errors_custom_m():
    """Test case with custom m (degrees of freedom)."""

    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2.5, 5.5, 6.8, 9.2])
    
    m = 3
    
    result = calculate_errors(y_true, y_pred, m)

    errors = y_true - y_pred
    sse = np.sum(errors ** 2)
    mse = sse / m
    rmse = np.sqrt(mse)
    
    assert np.isclose(result["Sum of Squared Errors"], sse)
    assert np.isclose(result["Mean Squared Error"], mse)
    assert np.isclose(result["Root Mean Squared Error"], rmse)


def test_calculate_errors_shape_mismatch():
    """Test invalid case."""

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    
    with pytest.raises(ValueError):
        calculate_errors(y_true, y_pred)


# ----- variance_inflation_factor() -----


def test_variance_inflation_factor_basic():
    """Test basic functionality: column names, shape, and feasible results."""

    # perfect collinearity (should result in inf VIF values)
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ])

    var_names = ["Test1", "Test2"]

    result = variance_inflation_factor(X, var_names)
    
    assert all(col in result.columns for col in ["Coefficient", "R-squared", "VIF"])
    assert result.shape == (2, 3)
    assert np.all(np.isposinf(result["VIF"]))
    assert all(col in var_names for col in result["Coefficient"].unique())


def test_variance_inflation_factor_single_predictor():
    """Test edge case (single predictor)."""

    X = np.array([
        [1],
        [2],
        [3]
    ])
    
    result = variance_inflation_factor(X)
    
    assert result["VIF"].iloc[0] == 1.0


def test_variance_inflation_factor_empty_input():
    """Test empty array contents."""

    with pytest.raises(ValueError):
        X = np.empty((0, 0))
        variance_inflation_factor(X)


def test_variance_inflation_factor_invalid_shape():
    """Test invalid array shape."""

    with pytest.raises(ValueError):
        X = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        variance_inflation_factor(X)
