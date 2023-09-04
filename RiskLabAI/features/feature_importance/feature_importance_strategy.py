from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List

class FeatureImportanceStrategy(ABC):
    @abstractmethod
    def calculate_importance(self, *args, **kwargs) -> pd.DataFrame:
        pass