# Testing

All tests are contained within `tests` directories for each module. You can simply execute the `pytest` command from project root to run all unit tests.

```bash
pytest
```

**Notes on Test Coverage:**
- Plotting functions from `turtles.plotting` are tested, but plotting methods in GLM classes 
(like `MLR`) are ignored. Those class methods are essentially just wrappers around `matplotlib` 
and `turtles.plotting` functions.
- `GLM` class methods that are meant to be implemented by child classes are ignored.
