"""
Main controller for processing tick data into bars.

This class reads data in batches from CSVs or DataFrames and
uses the BarsInitializerController to construct bars based on
a specified method.
"""
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Generator, Union, Dict, Any, List

from RiskLabAI.controller.bars_initializer import BarsInitializerController
from RiskLabAI.data.structures.abstract_bars import AbstractBars
from RiskLabAI.utils.constants import (
    DATE_TIME, TICK_NUMBER, OPEN_PRICE, HIGH_PRICE,
    LOW_PRICE, CLOSE_PRICE, CUMULATIVE_VOLUME,
    CUMULATIVE_BUY_VOLUME, CUMULATIVE_SELL_VOLUME,
    CUMULATIVE_TICKS, CUMULATIVE_DOLLAR, THRESHOLD
)

# Define the bar column schema
BAR_COLUMNS = [
    DATE_TIME, TICK_NUMBER, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE,
    CUMULATIVE_VOLUME, CUMULATIVE_BUY_VOLUME, CUMULATIVE_SELL_VOLUME,
    CUMULATIVE_TICKS, CUMULATIVE_DOLLAR, THRESHOLD
]

class Controller:
    """
    Controller for initializing and processing bars from data sources.
    """

    def __init__(self) -> None:
        """
        Initializes the Controller and its internal BarsInitializerController.
        """
        self.bars_initializer = BarsInitializerController()

    def handle_input_command(
        self,
        method_name: str,
        method_arguments: Dict[str, Any],
        input_data: Union[str, pd.DataFrame],
        output_path: Optional[str] = None,
        batch_size: int = 1_000_000
    ) -> pd.DataFrame:
        """
        Handles the command to initialize a bar generator and process data.
        (Parameters same as original)
        """
        # 1. Initialize the bar generator
        try:
            initializer_method = self.bars_initializer.method_name_to_method[method_name]
        except KeyError:
            print(f"Error: Bar method '{method_name}' not found.")
            valid_methods = list(self.bars_initializer.method_name_to_method.keys())
            print(f"Valid methods are: {valid_methods}")
            return pd.DataFrame(columns=BAR_COLUMNS)
            
        try:
            bar_generator: AbstractBars = initializer_method(**method_arguments)
        except TypeError as e:
            print(f"Error initializing bar method '{method_name}' with arguments {method_arguments}.")
            print(f"TypeError: {e}")
            return pd.DataFrame(columns=BAR_COLUMNS)

        # 2. Get the correct batch generator
        if isinstance(input_data, str):
            data_generator = self.read_batches_from_string(input_data, batch_size)
        elif isinstance(input_data, pd.DataFrame):
            data_generator = self.read_batches_from_dataframe(input_data, batch_size)
        else:
            raise TypeError("input_data must be a string (path) or pd.DataFrame")

        all_bars: List[List[Any]] = []
        
        # 3. Process data in batches
        print("Processing data in batches...")
        try:
            for data_batch in data_generator:
                # We assume data is [datetime, price, volume]
                # .values returns a NumPy array, which is an efficient iterable
                bars = bar_generator.construct_bars_from_data(data=data_batch.values)
                all_bars.extend(bars)
        except Exception as e:
            print(f"Error during bar construction: {e}")
            print("Returning DataFrame with bars constructed so far.")
            # Continue to return whatever was processed
            
        print(f"Done. Constructed {len(all_bars)} bars.")

        # 4. Create final DataFrame
        bars_df = pd.DataFrame(all_bars, columns=BAR_COLUMNS)

        if output_path:
            print(f"Saving bars to {output_path}...")
            try:
                bars_df.to_csv(output_path, index=False)
                print("Save complete.")
            except Exception as e:
                print(f"Error saving file to {output_path}: {e}")

        return bars_df

    @staticmethod
    def read_batches_from_string(
        input_path: str,
        batch_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Reads data in batches from a CSV file efficiently.

        Parameters
        ----------
        input_path : str
            File path to read from.
        batch_size : int
            Size of each batch.

        Yields
        -------
        Generator[pd.DataFrame, None, None]
            A generator yielding batches of data.
        """
        try:
            # Use a generator to read the file in chunks
            # This is more memory-efficient and avoids reading the file twice.
            for batch in pd.read_csv(
                input_path, chunksize=batch_size, parse_dates=[0]
            ):
                yield batch
        except FileNotFoundError:
            print(f"Error: File not found at {input_path}")
            return
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file {input_path}: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while reading {input_path}: {e}")
            return


    @staticmethod
    def read_batches_from_dataframe(
        input_data: pd.DataFrame,
        batch_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Reads data in batches from a DataFrame.
        """
        n_rows = input_data.shape[0]
        if n_rows == 0:
            return
            
        for start_row in range(0, n_rows, batch_size):
            end_row = min(start_row + batch_size, n_rows)
            yield input_data.iloc[start_row:end_row]