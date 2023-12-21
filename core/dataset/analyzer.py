import os
import json
import pandas as pd
import argparse
from pandas.api.types import is_numeric_dtype, is_categorical_dtype


class DatasetAnalyzer():
    """
    A class for preprocessing datasets for machine learning tasks.
    """

    def __init__(self, args: argparse.Namespace, target: str = None, task_type: str = None) -> None:
        """
        Initializes the DataPreprocessor with command-line arguments, target column, and task type.
        """
        self.args = args
        self.data_info = {"dataset_info": {"length": {}, "feature": {}}}
        self.target = target
        self.task_type = task_type
        self.data_file_path = os.path.join(args.data_dir, args.dataset)
        self.all_data_info = self._load_dataset_info()
        self.raw_data = self._load_data()

    def _load_dataset_info(self) -> dict:
        """
        Loads dataset information from a JSON file, creating a new one if it doesn't exist.
        """
        info_path = os.path.join(self.args.data_dir, 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def _save_info(self) -> None:
        """
        Saves the dataset information to a JSON file.
        """
        self.all_data_info[self.args.dataset] = self.data_info
        with open(os.path.join(self.args.data_dir, 'info.json'), 'w') as f:
            json.dump(self.all_data_info, f, indent=4)

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from a file. Supports CSV, TXT, and Excel formats.
        """
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(
                f"Dataset {self.args.dataset} not found in {self.args.data_dir}")

        if self.data_file_path.endswith('.csv'):
            data = pd.read_csv(self.data_file_path)
        elif self.data_file_path.endswith('.txt'):
            data = pd.read_table(self.data_file_path)
        elif self.data_file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(self.data_file_path)
        else:
            raise ValueError(f"Dataset {self.args.dataset} is not supported.")

        self._process_data(data)
        return data

    def _process_data(self, data: pd.DataFrame) -> None:
        """
        Processes the data by cleaning and inferring necessary information.
        """
        self._update_target_and_task_type(data)
        self._clean_data(data)
        self._update_info_dict(data)

    def _update_target_and_task_type(self, data: pd.DataFrame) -> None:
        """
        Updates the target and task type based on the data if they are not specified.
        """
        if self.target is None:
            self.target = self._infer_target(data)
        if self.task_type is None:
            self.task_type = self._infer_task_type(data)

    def _clean_data(self, data: pd.DataFrame) -> None:
        """
        Cleans the data by dropping empty rows and columns, and rows with missing target values.
        """
        data.dropna(axis=0, how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        data.dropna(subset=[self.target], inplace=True)

    def _update_info_dict(self, data: pd.DataFrame) -> None:
        """
        Updates the information dictionary with details about the dataset.
        """
        self.data_info['dataset_info']['length'] = {
            'rawdata_num_rows': len(data),
            'rawdata_num_cols': len(data.columns),
            'cleaned_num_rows': len(data.dropna(axis=0, how='all')),
            'cleaned_num_cols': len(data.dropna(axis=1, how='all')),
            'cleaned_num_rows_without_target': len(data.dropna(subset=[self.target]))
        }
        self.data_info['target'] = self.target
        self.data_info['task_type'] = self.task_type

    def _infer_target(self, data: pd.DataFrame) -> str:
        """
        Infers the target column, selecting the last column by default.
        """
        return data.columns[-1]

    def _infer_task_type(self, data: pd.DataFrame) -> str:
        """
        Infers the type of machine learning task based on the target column.
        """
        target_data = data[self.target]
        return 'classification' if target_data.dtype == 'object' or target_data.nunique() < 10 else 'regression'

    def get_feature_info(self) -> None:
        """
        Analyzes the dataset to identify and categorize features into numeric, categorical, and text types.
        """
        analysis_data = self.raw_data.drop(columns=[self.target])

        numeric_features, categorical_features, text_features = [], [], []
        for col in analysis_data.columns:
            non_na_values = analysis_data[col].dropna()
            unique_count = non_na_values.nunique()
            total_count = len(analysis_data[col])

            numeric_threshold = 0.05 * total_count
            category_threshold = 20

            if is_numeric_dtype(non_na_values):
                if unique_count <= category_threshold:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            elif is_categorical_dtype(non_na_values) or (non_na_values.dtype == object and unique_count <= numeric_threshold):
                categorical_features.append(col)
            else:
                text_features.append(col)

        self.data_info['dataset_info']['feature'] = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'text_features': text_features
        }
        self.data_info['dataset_info']['length'].update({
            'num_numeric_features': len(numeric_features),
            'num_categorical_features': len(categorical_features),
            'num_text_features': len(text_features)
        })

    def analyze_and_validate_dataset_info(self) -> None:
        """
        Analyzes the dataset, updates feature information, and validates the consistency of dataset information.
        """
        self.get_feature_info()

        existing_info = self.all_data_info.get(self.args.dataset)
        if existing_info:
            print(f"Dataset {self.args.dataset} has been analyzed before.")
            if existing_info == self.data_info:
                print("The dataset info is the same as before.")
            else:
                raise ValueError(
                    "The dataset info is different from the previous one.")
        else:
            print(f"Dataset {self.args.dataset} has not been analyzed before.")
            self._save_info()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tabular Data Predication (UTB)")
    parser.add_argument("--data_dir", type=str,
                        default="/home/qychen/workspace/UTB/data")
    parser.add_argument("-ds", "--dataset", type=str, default="test.csv")
    args = parser.parse_args()

    data_preprocessor = DatasetAnalyzer(args, target='Survived')
    data_preprocessor.analyze_and_validate_dataset_info()
