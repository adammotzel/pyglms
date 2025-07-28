"""
Unit tests for GLM validation.
"""

import pytest

from .._validation import _validate_init


def test_failures():
    """Test cases that should fail."""

    with pytest.raises(ValueError):
        _validate_init(
            max_iter=1000, 
            learning_rate=0.1, 
            tolerance=0.01, 
            method="test", 
            beta_momentum=1.9
        )

    with pytest.raises(ValueError):
        _validate_init(
            max_iter=-5, 
            learning_rate=0.1, 
            tolerance=0.01, 
            method="lbfgs", 
            beta_momentum=1.0
        )

    with pytest.raises(ValueError):
        _validate_init(
            max_iter=1000, 
            learning_rate=-0.1, 
            tolerance=0.01, 
            method="lbfgs", 
            beta_momentum=0.6
        )

    with pytest.raises(ValueError):
        _validate_init(
            max_iter=1000, 
            learning_rate=0.1, 
            tolerance=1, 
            method="lbfgs", 
            beta_momentum=0.6
        )

    with pytest.raises(ValueError):
        _validate_init(
            max_iter=0, 
            learning_rate=0, 
            tolerance=0, 
            method="lbfgs", 
            beta_momentum=0
        )

    with pytest.raises(ValueError):
        _validate_init(
            max_iter=0.001, 
            learning_rate=0.001, 
            tolerance=0.999, 
            method="grad", 
            beta_momentum=0
        )


def test_valid():
    """Test valid cases."""

    _validate_init(
        max_iter=1000, 
        learning_rate=0.1, 
        tolerance=0.01, 
        method="lbfgs", 
        beta_momentum=0.6
    )

    _validate_init(
        max_iter=1, 
        learning_rate=0.001, 
        tolerance=0.999, 
        method="grad", 
        beta_momentum=0
    )
