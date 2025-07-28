"""
Unit tests for functions from the preprocess module.
"""

import pandas as pd
import numpy as np

from ...preprocess import one_hot_encode


def test_one_hot_encode_basic():
    """Test basic functionality."""

    data = {
        "color": ["red", "blue", "green"],
        "size": ["M", "L", "M"]
    }
    df = pd.DataFrame(data)

    expected_df = pd.DataFrame({
        "color_green": [0, 0, 1],
        "color_red": [1, 0, 0],
        "size_M": [1, 0, 1]
    })

    result = one_hot_encode(
        df, 
        columns=["color", "size"], 
        drop_first=True, 
        return_df=True
    )

    pd.testing.assert_frame_equal(result, expected_df)


def test_one_hot_encode_return_df_false():
    """Test return type when `return_df` is False."""

    data = {
        "color": ["red", "blue", "green"],
        "size": ["M", "L", "M"]
    }
    df = pd.DataFrame(data)

    result, column_names = one_hot_encode(
        df, 
        columns=["color", "size"], 
        drop_first=True, 
        return_df=False
    )

    expected_array = np.array([
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1]
    ])
    expected_columns = [
        "color_green", 
        "color_red", 
        "size_M"
    ]

    np.testing.assert_array_equal(result, expected_array)
    assert column_names == expected_columns


def test_one_hot_encode_drop_first():
    """Test if the "drop_first" option works correctly."""

    data = {
        "color": ["red", "blue", "green"],
        "size": ["M", "L", "M"]
    }
    df = pd.DataFrame(data)

    expected_df_drop_first = pd.DataFrame({
        "color_blue": [0, 1, 0],
        "color_green": [0, 0, 1],
        "color_red": [1, 0, 0],
        "size_L": [0, 1, 0],
        "size_M": [1, 0, 1]
    })

    result = one_hot_encode(
        df, 
        columns=["color", "size"], 
        drop_first=False, 
        return_df=True
    )

    pd.testing.assert_frame_equal(result, expected_df_drop_first)


def test_one_hot_encode_return_df_false_no_drop_first():
    """
    Test if the function works when `return_df` is False and `drop_first` is False.
    """
    data = {
        "color": ["red", "blue", "green"],
        "size": ["M", "L", "M"]
    }
    df = pd.DataFrame(data)

    result, column_names = one_hot_encode(
        df, 
        columns=["color", "size"], 
        drop_first=False, 
        return_df=False
    )

    expected_array = np.array([
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1]
    ])
    expected_columns = ["color_blue", "color_green", "color_red", "size_L", "size_M"]

    np.testing.assert_array_equal(result, expected_array)
    assert column_names == expected_columns


def test_one_hot_encode_no_categorical_columns():
    """Test the output when there are no categorical columns to encode."""

    data = {
        "age": [23, 45, 34],
        "income": [50000, 60000, 55000]
    }
    df = pd.DataFrame(data)

    result = one_hot_encode(
        df, 
        columns=[], 
        drop_first=True, 
        return_df=True
    )

    pd.testing.assert_frame_equal(result, df)
