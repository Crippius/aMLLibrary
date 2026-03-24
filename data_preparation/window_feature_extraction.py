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

import numpy as np
import pandas as pd

import data_preparation.data_preparation
import regression_inputs


class WindowFeatureExtraction(data_preparation.data_preparation.DataPreparation):
    """
    Step that aggregates raw window step-columns into statistical features

    Consumes the output of TemporalWindowing (columns named {col}_step_0 through
    {col}_step_{w-1}) and replaces them with one aggregated column per
    (original column, feature) pair, e.g. cpu_mean, cpu_std, mem_slope.
    Output is a standard flat DataFrame consumed unchanged by the rest of the
    existing pipeline.

    Config keys read from [WindowFeatures]:
        features  list of str  features to compute per column
                  Supported: mean, std, min, max, range, slope, skewness, kurtosis
                  [default: ['mean', 'std', 'min', 'max']]

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Extract statistical features and return a new RegressionInputs
    """

    def get_name(self):
        """
        Return "WindowFeatureExtraction"

        Returns
        -------
        string
            The name of this step
        """
        return "WindowFeatureExtraction"

    def process(self, inputs):
        """
        Main method of the class which performs the actual feature extraction

        Parameters
        ----------
        inputs: RegressionInputs
            The windowed data produced by TemporalWindowing
        """
        wf = self._campaign_configuration.get('WindowFeatures', {})
        features = wf.get('features', ['mean', 'std', 'min', 'max'])

        y_col = self._campaign_configuration['General']['y']
        window_size = self._campaign_configuration['DataPreparation']['window_size']

        df = inputs.data

        # Discover original column names from the step-expanded columns
        original_cols = []
        seen = set()
        for c in df.columns:
            if '_step_' in c:
                base = c[:c.rfind('_step_')]
                if base not in seen:
                    original_cols.append(base)
                    seen.add(base)

        new_df = pd.DataFrame(index=df.index)

        # Pre-compute centered x deviations for slope (constant for fixed window_size)
        x = np.arange(window_size, dtype=float)
        x_dev = x - x.mean()
        ss_xx = (x_dev ** 2).sum()

        for col in original_cols:
            step_col_names = [f'{col}_step_{i}' for i in range(window_size)]
            vals = df[step_col_names].values  # shape (n_windows, window_size)

            for feat in features:
                if feat == 'mean':
                    new_df[f'{col}_mean'] = vals.mean(axis=1)
                elif feat == 'std':
                    new_df[f'{col}_std'] = vals.std(axis=1)
                elif feat == 'min':
                    new_df[f'{col}_min'] = vals.min(axis=1)
                elif feat == 'max':
                    new_df[f'{col}_max'] = vals.max(axis=1)
                elif feat == 'range':
                    new_df[f'{col}_range'] = vals.max(axis=1) - vals.min(axis=1)
                elif feat == 'slope':
                    if ss_xx != 0:
                        y_means = vals.mean(axis=1, keepdims=True)
                        ss_xy = ((vals - y_means) * x_dev).sum(axis=1)
                        new_df[f'{col}_slope'] = ss_xy / ss_xx
                    else:
                        new_df[f'{col}_slope'] = 0.0
                elif feat == 'skewness':
                    from scipy.stats import skew
                    new_df[f'{col}_skewness'] = skew(vals, axis=1)
                elif feat == 'kurtosis':
                    from scipy.stats import kurtosis
                    new_df[f'{col}_kurtosis'] = kurtosis(vals, axis=1)
                else:
                    self._logger.warning("Unknown window feature '%s' - skipped", feat)

        # Preserve y column if present
        if y_col in df.columns:
            new_df[y_col] = df[y_col].values

        new_x_cols = [c for c in new_df.columns if c != y_col]

        # Update feature names so downstream steps (column selection, etc.) see the new columns
        if 'Features' in self._campaign_configuration:
            self._campaign_configuration['Features']['Original_feature_names'] = new_x_cols

        self._logger.info("WindowFeatureExtraction: produced %d features for %d windows",
                          len(new_x_cols), len(new_df))

        return regression_inputs.RegressionInputs(new_df, inputs.inputs_split, new_x_cols, y_col)
