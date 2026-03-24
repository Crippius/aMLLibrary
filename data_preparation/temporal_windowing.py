"""
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

import sys

import pandas as pd

import data_preparation.data_preparation
import regression_inputs


class TemporalWindowing(data_preparation.data_preparation.DataPreparation):
    """
    Step that converts a time-ordered DataFrame into a windowed tabular format

    For each window of size w sliding over the time-ordered data (optionally
    grouped by a series identifier), one row is produced containing the raw
    values of every X column at each time step, named {col}_step_0 through
    {col}_step_{w-1}.  The target y is extracted from the y column at the
    position specified by y_window_position.

    Config keys read from [DataPreparation]:
        window_size      (int)  number of time steps per window  [required]
        stride           (int)  step between consecutive windows [default: 1]
        series_id_column (str)  group column for multi-series    [optional]

    Config keys read from [WindowFeatures]:
        y_window_position  "last" | "first" | "mean"            [default: "last"]

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Apply the sliding window and return a new RegressionInputs
    """

    def get_name(self):
        """
        Return "TemporalWindowing"

        Returns
        -------
        string
            The name of this step
        """
        return "TemporalWindowing"

    def process(self, inputs):
        """
        Main method of the class which performs the actual windowing

        Parameters
        ----------
        inputs: RegressionInputs
            The time-ordered data to be windowed
        """
        dp = self._campaign_configuration['DataPreparation']
        window_size = dp['window_size']
        stride = int(dp.get('stride', 1))
        series_id_col = dp.get('series_id_column', None)

        y_col = self._campaign_configuration['General']['y']
        y_window_position = self._campaign_configuration.get('WindowFeatures', {}).get('y_window_position', 'last')

        df = inputs.data.reset_index(drop=True)
        y_present = y_col in df.columns

        # Determine iteration over groups (multi-series or single series)
        if series_id_col and series_id_col in df.columns:
            group_iter = [(gid, gdf.reset_index(drop=True)) for gid, gdf in df.groupby(series_id_col, sort=False)]
        else:
            group_iter = [(None, df)]

        rows = []
        for _, group_df in group_iter:
            n = len(group_df)
            for start in range(0, n - window_size + 1, stride):
                window = group_df.iloc[start:start + window_size]
                row = {}

                for col in group_df.columns:
                    if series_id_col and col == series_id_col:
                        continue
                    if col == y_col:
                        if y_present:
                            vals = window[col]
                            if y_window_position == 'last':
                                row[y_col] = vals.iloc[-1]
                            elif y_window_position == 'first':
                                row[y_col] = vals.iloc[0]
                            elif y_window_position == 'mean':
                                row[y_col] = vals.mean()
                            else:
                                self._logger.error("Unknown y_window_position: %s", y_window_position)
                                sys.exit(1)
                    else:
                        for i, val in enumerate(window[col].values):
                            row[f'{col}_step_{i}'] = val

                rows.append(row)

        new_df = pd.DataFrame(rows).reset_index(drop=True)
        new_x_cols = [c for c in new_df.columns if c != y_col]
        new_inputs_split = {
            'training': new_df.index.values.tolist(),
            'all': new_df.index.values.tolist(),
        }

        self._logger.info("TemporalWindowing: %d windows from %d time steps (window_size=%d, stride=%d)",
                          len(new_df), len(df), window_size, stride)

        return regression_inputs.RegressionInputs(new_df, new_inputs_split, new_x_cols, y_col)
