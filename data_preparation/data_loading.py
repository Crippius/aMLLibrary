"""
Copyright 2019 Marco Lattuada
Copyright 2026 Tommaso Crippa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys

import pandas as pd

import data_preparation.data_preparation
import regression_inputs


class DataLoading(data_preparation.data_preparation.DataPreparation):
    """
    Step which load data from csv

    This step is the first to be executed in the whole flow

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Read the data
    """

    def get_name(self):
        """
        Return "DataLoading"

        Returns
        string
            The name of this step
        """
        return "DataLoading"

    def process(self, inputs):
        """
        Main method of the class which performs the actual load and return a RegressionInputs

        In the created RegressionInputs, training set is put equal to the whole input dataset

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        input_path = self._campaign_configuration['DataPreparation']['input_path']
        if isinstance(input_path, str):
            self._logger.info("Input reading: %s", input_path)
            if not os.path.exists(input_path):
                self._logger.error("%s not found", input_path)
                sys.exit(-1)
            data_frame = pd.read_csv(input_path)
        elif isinstance(input_path, pd.DataFrame):
            data_frame = input_path.copy()
        else:
            self._logger.error("input_path must be a path string to a dataset or a pandas.DataFrame")
            sys.exit(1)

        # Sort by time_column and drop it (activates time series adapter)
        time_col = self._campaign_configuration['DataPreparation'].get('time_column', None)
        series_id_col = self._campaign_configuration['DataPreparation'].get('series_id_column', None)
        if time_col:
            if time_col not in data_frame.columns:
                self._logger.error("time_column '%s' not found in dataset", time_col)
                sys.exit(1)
            data_frame[time_col] = pd.to_datetime(data_frame[time_col])
            data_frame = data_frame.sort_values(by=time_col).reset_index(drop=True)
            data_frame = data_frame.drop(columns=[time_col])

        y_col = self._campaign_configuration['General']['y']
        self._campaign_configuration['Features'] = {}
        self._campaign_configuration['Features']['Original_feature_names'] = []
        for column_name in data_frame.columns.values:
            # Exclude y column and series_id_column from feature names
            if column_name == y_col:
                continue
            if series_id_col and column_name == series_id_col:
                continue
            self._campaign_configuration['Features']['Original_feature_names'].append(column_name)

        inputs_split = {}
        inputs_split["training"] = data_frame.index.values.tolist()
        inputs_split["all"] = inputs_split["training"].copy()

        output = regression_inputs.RegressionInputs(data_frame, inputs_split, self._campaign_configuration['Features']['Original_feature_names'], y_col)
        return output
