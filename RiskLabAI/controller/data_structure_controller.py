import csv
from math import floor
from typing import Iterable, Optional, Generator, Union

import pandas as pd

from RiskLabAI.controller.bars_initializer import BarsInitializerController
from RiskLabAI.data.structures.abstract_bars import AbstractBars
from RiskLabAI.utils.constants import (
    DATE_TIME, TICK_NUMBER, OPEN_PRICE, HIGH_PRICE,
    LOW_PRICE, CLOSE_PRICE, CUMULATIVE_VOLUME,
    CUMULATIVE_BUY_VOLUME, CUMULATIVE_SELL_VOLUME,
    CUMULATIVE_TICKS, CUMULATIVE_DOLLAR, THRESHOLD
)


class Controller:
    def __init__(self) -> None:
        self.bars_initializer = BarsInitializerController()

    def handle_input_command(
            self,
            method_name: str,
            method_arguments: dict,
            input_data: Union[str, pd.DataFrame],
            output_path: Optional[str] = None,
            batch_size: int = 1_000_000
    ) -> pd.DataFrame:
        """
        Handles the input command to initialize bars and run on batches.

        :param method_name: Name of the method to call
        :param method_arguments: Arguments for the method
        :param input_data: Input data as a DataFrame or string path
        :param output_path: Optional path to save results as CSV
        :param batch_size: Size of each batch to process
        :return: DataFrame of aggregated bars
        """
        method = self.bars_initializer.method_name_to_method[method_name]
        initialized_bars = method(**method_arguments)

        return self.run_on_batches(
            initialized_bars=initialized_bars,
            input_data=input_data,
            batch_size=batch_size,
            output_path=output_path
        )

    def run_on_batches(
            self,
            initialized_bars: AbstractBars,
            input_data: Union[str, pd.DataFrame],
            batch_size: int,
            output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Runs the initialized bars on batches of data.

        :param initialized_bars: Initialized bars object
        :param input_data: Input data as DataFrame or string path
        :param batch_size: Size of each batch to process
        :param output_path: Optional path to save results as CSV
        :return: DataFrame of aggregated bars
        """
        save_to_csv = output_path is not None

        if save_to_csv:
            with open(output_path, 'w'):
                pass
            add_header = True

        column_names = [
            DATE_TIME, TICK_NUMBER, OPEN_PRICE, HIGH_PRICE,
            LOW_PRICE, CLOSE_PRICE, CUMULATIVE_VOLUME,
            CUMULATIVE_BUY_VOLUME, CUMULATIVE_SELL_VOLUME,
            CUMULATIVE_TICKS, CUMULATIVE_DOLLAR, THRESHOLD
        ]

        aggregated_bars = []

        if isinstance(input_data, str):
            batches_generator = self.read_batches_from_string(
                input_data, batch_size)
        elif isinstance(input_data, pd.DataFrame):
            batches_generator = self.read_batches_from_dataframe(
                input_data, batch_size)
        else:
            raise ValueError('Input data type should be DataFrame or string.')

        for k, batch in enumerate(batches_generator):
            print(f"Processing batch {k} with size {batch.shape[0]}")

            batch_bars = self.construct_bars_from_batch(
                initialized_bars, batch)

            aggregated_bars.extend(batch_bars)

            if save_to_csv:
                pd.DataFrame(
                    batch_bars, columns=column_names
                ).to_csv(output_path, header=add_header, index=False, mode='a')
                add_header = False

        return pd.DataFrame(aggregated_bars, columns=column_names)

    @staticmethod
    def construct_bars_from_batch(
            bars: AbstractBars,
            data: pd.DataFrame
    ) -> list:
        """
        Construct bars from a single batch of data.

        :param bars: Initialized bars object
        :param data: Data for this batch as a DataFrame
        :return: List of constructed bars
        """
        return bars.construct_bars_from_data(data=data.values)

    @staticmethod
    def read_batches_from_string(
            input_path: str,
            batch_size: int
    ) -> Generator:
        """
        Reads data in batches from a CSV file.

        :param input_path: File path to read from
        :param batch_size: Size of each batch
        :return: Generator yielding batches of data
        """
        n_rows = sum(1 for _ in csv.reader(open(input_path)))
        n_batches = max(1, floor(n_rows / batch_size))

        if n_batches == 1:
            yield pd.read_csv(input_path, parse_dates=[0])
        else:
            for batch in pd.read_csv(input_path, chunksize=batch_size, parse_dates=[0]):
                yield batch

    @staticmethod
    def read_batches_from_dataframe(
            input_data: pd.DataFrame,
            batch_size: int
    ) -> Generator:
        """
        Reads data in batches from a DataFrame.

        :param input_data: DataFrame to read from
        :param batch_size: Size of each batch
        :return: Generator yielding batches of data
        """
        n_rows = input_data.shape[0]
        n_batches = max(1, floor(n_rows / batch_size))

        if n_batches == 1:
            yield input_data
        else:
            for k in range(n_batches):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, n_rows)
                yield input_data.iloc[start_idx:end_idx].copy()
