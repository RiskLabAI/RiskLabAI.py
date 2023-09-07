from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List


class FeatureImportanceStrategy(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs) -> pd.DataFrame:
        pass
