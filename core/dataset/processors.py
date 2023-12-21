import pandas as pd
from typing import List, Union
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class DataPreprocessor:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors if preprocessors else []

    def add_preprocessor(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def preprocess_data(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)
        return data


class NaNPreprocessor(BasePreprocessor):
    def __init__(self, strategy: str = 'auto', fill_value: Union[float, str] = None,
                 columns: List[str] = None):
        """
        Initialize NaNPreprocessor with a specified strategy.
        :param strategy: Strategy to handle NaN values ('mean', 'median', 'mode', 'value', 'drop_row', 'drop_column', 'auto').
        :param fill_value: Value used for 'value' strategy.
        :param columns: List of columns to apply NaN processing. If None, applies to all columns.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            columns_to_process = self.columns
        else:
            columns_to_process = data.columns

        for column in columns_to_process:
            if self.strategy == 'auto':
                self._auto_strategy(data, column)
            elif self.strategy == 'mean':
                data[column].fillna(data[column].mean(), inplace=True)
            elif self.strategy == 'median':
                data[column].fillna(data[column].median(), inplace=True)
            elif self.strategy == 'mode':
                data[column].fillna(data[column].mode()[0], inplace=True)
            elif self.strategy == 'value':
                data[column].fillna(self.fill_value, inplace=True)
            elif self.strategy == 'drop_row':
                data.dropna(subset=[column], inplace=True)
            elif self.strategy == 'drop_column':
                data.drop(columns=[column], inplace=True)
            else:
                raise ValueError(f"Strategy {self.strategy} not recognized")

        return data

    def _auto_strategy(self, data: pd.DataFrame, column: str):
        """
        Automatically select the strategy based on the column type.
        """
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)


if __name__=="__main__":
    data_preprocessor = DataPreprocessor()

    # 添加所需的预处理器
    data_preprocessor.add_preprocessor(NaNPreprocessor())

    # 执行预处理
    processed_data = data_preprocessor.preprocess_data(raw_data)
