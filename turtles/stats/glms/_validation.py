"""
Validation for GLM class instantiation.
"""

import json
from typing import Literal, List

from pydantic import BaseModel, Field, ValidationError


class GLMInit(BaseModel):
    """Validates the arguments passed during GLM instantiation."""

    max_iter: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    tolerance: float = Field(gt=0, lt=1)
    beta_momentum: float = Field(ge=0)
    method: Literal["newton", "grad", "lbfgs"]


def _validate_init(**kwargs):
    """
    Validate arguments passed during GLM instatiation. If invalid, 
    calls `_raise_error()` with the error message to raise a ValueError.

    Parameters
    ----------
    **kwargs
        The key:value pairs to pass to the GLMInit class.

    Returns
    -------
    None
    """

    errors = None
    
    try:
        GLMInit(**kwargs)
    except ValidationError as ve:
        errors = ve.errors()
    
    _raise_error(errors)


def _raise_error(errors: List[dict]):
    """
    Raise a ValueError if an error message is passed.

    Parameters
    ----------
    errors : List[dict]
        A list of errors collected from a pydantic ValidationError.

    Raises
    ------
    ValueError
        If `errors` contains error messages.

    Returns
    -------
    None
    """

    if errors:
        errs = {
            "errors": [
                {
                    "parameter": error["loc"][0],
                    "error_message": error["msg"],
                    "argument_received": error["input"]
                } for error in errors
            ]
        }

        dumps = json.dumps(errs, indent=2)

        raise ValueError(
            f"Invalid arguments passed during GLM instantation. Errors: \n\n{dumps}"
        )
    