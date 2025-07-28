"""
Unit tests for functions from the plotting module.
"""

from unittest.mock import patch

import numpy as np

from ...plotting import (
    plot_y_vs_x
)


def test_plot_valid_inputs():
    """Test if `plt.show()` is called once."""

    x = np.array([1, 2, 3, 4])
    y = np.array([2, 4, 6, 8])

    # patch plt.show to avoid displaying the plot during testing
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_y_vs_x(x, y)
        mock_show.assert_called_once() 


def test_default_labels_and_title():
    """Test if the function uses correct default labels and title."""

    x = np.array([1, 2, 3, 4])
    y = np.array([2, 4, 6, 8])
    
    with patch("matplotlib.pyplot.show") as mock_show, \
        patch("matplotlib.pyplot.title") as mock_title, \
        patch("matplotlib.pyplot.xlabel") as mock_xlabel, \
        patch("matplotlib.pyplot.ylabel") as mock_ylabel:

        plot_y_vs_x(x, y)

        mock_title.assert_called_with("Dependent vs. Independent")
        mock_xlabel.assert_called_with("Independent")
        mock_ylabel.assert_called_with("Dependent")


def test_custom_labels_and_title():
    """Test if custom title and labels work."""

    x = np.array([1, 2, 3, 4])
    y = np.array([2, 4, 6, 8])
    title = "Custom Title"
    xlabel = "Custom X-Axis"
    ylabel = "Custom Y-Axis"

    with patch("matplotlib.pyplot.show") as mock_show, \
        patch("matplotlib.pyplot.title") as mock_title, \
        patch("matplotlib.pyplot.xlabel") as mock_xlabel, \
        patch("matplotlib.pyplot.ylabel") as mock_ylabel:

        plot_y_vs_x(x, y, title=title, xlabel=xlabel, ylabel=ylabel)

        mock_title.assert_called_with(title)
        mock_xlabel.assert_called_with(xlabel)
        mock_ylabel.assert_called_with(ylabel)
