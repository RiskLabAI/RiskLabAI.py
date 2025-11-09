"""
Controller to simplify the creation and use of cross-validators.
"""

from typing import Any

from .cross_validator_factory import CrossValidatorFactory
from .cross_validator_interface import CrossValidator

class CrossValidatorController:
    """
    Controller class to handle the cross-validation process.

    This class acts as a high-level interface, simplifying the
    creation and access to a specific cross-validator using the factory.
    """

    def __init__(
        self,
        validator_type: str,
        **kwargs: Any
    ):
        """
        Initializes the CrossValidatorController.

        Parameters
        ----------
        validator_type : str
            Type of cross-validator to create (e.g., 'kfold',
            'combinatorialpurged'). This is passed to the factory.
        **kwargs : Any
            Additional keyword arguments to be passed to the
            cross-validator's constructor.
        """
        self.cross_validator: CrossValidator = \
            CrossValidatorFactory.create_cross_validator(
                validator_type,
                **kwargs
            )

    def get_validator(self) -> CrossValidator:
        """
        Get the created cross-validator instance.

        Returns
        -------
        CrossValidator
            The underlying cross-validator instance.
        """
        return self.cross_validator