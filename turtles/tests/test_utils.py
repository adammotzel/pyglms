"""
Unit tests for functions from the _utils module.
"""

import pytest
import numpy as np

from .._utils import (
    _add_intercept,
    _validate_args,
    _shape_check
)


# ----- _validate_args() -----


def test_valid_input():
    """Test valid input, correct types."""

    X = np.array([1, 2, 3])
    Y = [1.5, 2.5, 3.5]

    _validate_args(
        {
            "X": (X, np.ndarray),
            "Y": (Y, list)
        }
    )


def test_invalid_type_X():
    """Test invalid input type."""

    X = [1, 2, 3]
    Y = [1.5, 2.5, 3.5]

    with pytest.raises(
        TypeError, 
        match="Parameter 'X' must be of type <class 'numpy.ndarray'>; received <class 'list'>"
    ):
        _validate_args(
            {
                "X": (X, np.ndarray),
                "Y": (Y, (int, float))
            }
        )


def test_invalid_type_Y():
    """Test invalid input type."""

    X = np.array([1, 2, 3])
    Y = "invalid_string"

    with pytest.raises(
        TypeError, 
        match="Parameter 'Y' must be one of the types \\(<class 'int'>, <class 'float'>\\); received <class 'str'>"
    ):
        _validate_args(
            {
                "X": (X, np.ndarray),
                "Y": (Y, (int, float))
            }
        )


def test_multiple_valid_types():
    """Test multiple valid types."""

    X = np.array([1, 2, 3])
    Y = 4 

    _validate_args(
        {
            "X": (X, np.ndarray),
            "Y": (Y, (int, float))
        }
    )


def test_multiple_invalid_types():
    """Test multiple invalid types."""

    X = 3.14
    Y = [1.5, 2.5, 3.5]

    with pytest.raises(
        TypeError, 
        match="Parameter 'X' must be of type <class 'numpy.ndarray'>; received <class 'float'>"
    ):
        
        _validate_args(
            {
                "X": (X, np.ndarray),
                "Y": (Y, (int, float))
            }
        )


# ----- _add_intercept() -----


def test_add_intercept_valid_input():
    """Test valid input."""

    X = np.array([[1, 2], [3, 4], [5, 6]])

    # Expected result: a column of ones added as intercept
    expected_output = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])

    result = _add_intercept(X)
    np.testing.assert_array_equal(result, expected_output)


def test_add_intercept_empty():
    """Test empty matrix."""

    X = np.array([]).reshape(0, 0)

    expected_output = np.array([]).reshape(0, 1)
    
    result = _add_intercept(X)
    np.testing.assert_array_equal(result, expected_output)


def test_add_intercept_single_row():
    """Test single row, multiple dimensions."""

    X = np.array([[1, 2, 3]])

    expected_output = np.array([[1, 1, 2, 3]])

    result = _add_intercept(X)
    np.testing.assert_array_equal(result, expected_output)


def test_add_intercept_single_feature():
    """Test multiple rows, single dimension."""

    X = np.array([[1], [2], [3]])

    expected_output = np.array([[1, 1], [1, 2], [1, 3]])

    result = _add_intercept(X)
    np.testing.assert_array_equal(result, expected_output)


def test_add_intercept_shape():
    """Test resulting shape."""

    X = np.array([[1, 2, 3], [4, 5, 6]])

    result = _add_intercept(X)

    assert result.shape == (2, 4)


# ----- _shape_check() -----


def test_shape_check_valid_2d():
    """Test valid case."""

    arr = np.array([[1, 2], [3, 4]])
    _shape_check(arr, "arr", dim=1)


def test_shape_check_empty_2d():
    """Test invalid case."""

    arr = np.array([[], []])
    with pytest.raises(ValueError):
        _shape_check(arr, "arr", dim=1)


def test_shape_check_invalid_1d():
    """Test invalid case."""

    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _shape_check(arr, "arr", dim=2)


def test_shape_check_invalid_2d():
    """Test invalid case."""

    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _shape_check(arr, "arr", dim=3)
