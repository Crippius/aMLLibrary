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

import custom_logger
import data_preparation.temporal_windowing
import data_preparation.window_feature_extraction
import regression_inputs
import regressor


class MTSRegressor(regressor.Regressor):
    """
    Regressor for multivariate time series data.

    Extends Regressor by applying the sliding-window feature
    extraction pipeline before any prediction, so callers pass raw
    time-ordered DataFrames and receive per-window predictions.

    Methods
    -------
    predict(inputs)
        Apply windowing to raw time series input, then delegate to the
        parent predict().

    get_true_y(df)
        Apply windowing to a full DataFrame (including y) and return the
        windowed y values, aligned with the predictions returned by predict().

    from_regressor(r)
        Class method — convert an existing Regressor into an MTSRegressor,
        copying all internal state.
    """

    @classmethod
    def from_regressor(cls, r):
        """
        Convert a trained Regressor into an MTSRegressor.

        Parameters
        ----------
        r: Regressor
            The regressor to convert (returned by ModelBuilding.process())

        Returns
        -------
        MTSRegressor
        """
        obj = cls.__new__(cls)
        obj.__dict__.update(r.__dict__.copy())
        obj._logger = custom_logger.getLogger(__name__)
        return obj

    def _apply_windowing(self, df):
        """
        Apply TemporalWindowing + WindowFeatureExtraction to df.

        Parameters
        ----------
        df: pandas.DataFrame
            Time-ordered DataFrame. May or may not contain the y column.

        Returns
        -------
        pandas.DataFrame
            Windowed feature DataFrame (one row per window).
        """
        y_col = self._campaign_configuration['General']['y']
        x_cols = [c for c in df.columns if c != y_col]
        ri = regression_inputs.RegressionInputs(
            df,
            {'training': df.index.tolist(), 'all': df.index.tolist()},
            x_cols,
            y_col,
        )
        ri = data_preparation.temporal_windowing.TemporalWindowing(
            self._campaign_configuration
        ).process(ri)
        ri = data_preparation.window_feature_extraction.WindowFeatureExtraction(
            self._campaign_configuration
        ).process(ri)
        return ri.data

    def predict(self, inputs):
        """
        Apply windowing to raw time series input, then predict.

        Parameters
        ----------
        inputs: pandas.DataFrame
            Time-ordered DataFrame of input metrics (y column excluded).
        """
        windowed = self._apply_windowing(inputs)
        self._logger.debug("MTSRegressor: windowed input to %d rows x %d cols", *windowed.shape)
        return super().predict(windowed)

    def get_true_y(self, df):
        """
        Return windowed y values aligned with the predictions from predict().

        Parameters
        ----------
        df: pandas.DataFrame
            Full time-ordered DataFrame including the y column.

        Returns
        -------
        pandas.Series
            One y value per window (at the configured y_window_position).
        """
        windowed = self._apply_windowing(df)
        y_col = self._campaign_configuration['General']['y']
        return windowed[y_col]
