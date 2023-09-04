from feature_importance_factory import FeatureImportanceFactory

class FeatureImportanceController:
    """
    Controller class to manage various feature importance strategies.

    To use this controller class:
        1. Initialize it with the type of feature importance strategy you want to use, along with any required parameters for that strategy.
        2. Call the `calculate` method to actually perform the feature importance calculation.

    For example:
    ```python
    # Initialize the controller with a 'ClusteredMDA' strategy
    controller = FeatureImportanceController('ClusteredMDA', classifier=my_classifier, clusters=my_clusters)

    # Calculate feature importance
    result = controller.calculate(my_x, my_y)
    ```
    """

    def __init__(self, strategy_type: str, **kwargs):
        """
        Initialize the controller with a specific feature importance strategy.

        :param strategy_type: The type of feature importance strategy to use.
        :param kwargs: Additional arguments to pass to the strategy class.
        """
        self.feature_importance_instance = FeatureImportanceFactory.create_feature_importance(strategy_type, **kwargs)

    def calculate(self, x, y, **kwargs):
        """
        Calculate feature importance based on the initialized strategy.

        :param x: Feature data.
        :param y: Target data.
        :param kwargs: Additional arguments to pass to the calculation method.
        :return: Feature importance result.
        """
        return self.feature_importance_instance.calculate(x, y, **kwargs)