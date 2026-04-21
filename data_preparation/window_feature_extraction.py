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
    Step that aggregates raw window step-columns into statistical features.

    Consumes the output of TemporalWindowing (columns named {col}_step_0 through
    {col}_step_{w-1}) and replaces them with one aggregated column per
    (original column, feature) pair.

    Config keys read from [WindowFeatures]:
        features  list   features to compute per column.
                  Each element is either a plain string (simple feature) or a
                  single-key dict (parameterized feature):

            ['mean', 'standard_deviation', 'minimum', 'maximum', 'range',
             'slope', 'skewness', 'kurtosis',
             {'quantile': {'q': 0.05}},
             {'quantile': {'q': 0.25}},
             {'autocorrelation': {'f_agg': 'mean',   'maxlag': 3}},
             {'autocorrelation': {'f_agg': 'median', 'maxlag': 3}},
             {'linear_trend': {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'}},
            ]

    """

    def get_name(self):
        return "WindowFeatureExtraction"

    def process(self, inputs):
        wf = self._campaign_configuration.get('WindowFeatures', {})
        raw_features = wf.get('features', ['mean', 'std', 'min', 'max'])
        features = self._normalize_features(raw_features)

        y_col = self._campaign_configuration['General']['y']
        window_size = self._campaign_configuration['DataPreparation']['window_size']

        df = inputs.data

        original_cols = []
        seen = set()
        for c in df.columns:
            if '_step_' in c:
                base = c[:c.rfind('_step_')]
                if base not in seen:
                    original_cols.append(base)
                    seen.add(base)

        new_df = pd.DataFrame(index=df.index)

        for col in original_cols:
            step_cols = [f'{col}_step_{i}' for i in range(window_size)]
            vals = df[step_cols].values 
            self._add_features(new_df, col, vals, window_size, features)

        if y_col in df.columns:
            new_df[y_col] = df[y_col].values

        new_x_cols = [c for c in new_df.columns if c != y_col]

        if 'Features' in self._campaign_configuration:
            self._campaign_configuration['Features']['Original_feature_names'] = new_x_cols

        self._logger.info(
            "WindowFeatureExtraction: produced %d features for %d windows",
            len(new_x_cols), len(new_df),
        )

        return regression_inputs.RegressionInputs(new_df, inputs.inputs_split, new_x_cols, y_col)

    def _normalize_features(self, raw):
        """Return a canonical dict {feature_name: None | list_of_param_dicts}."""
        out = {}
        for item in raw:
            if isinstance(item, str):
                if item not in out:
                    out[item] = None
            elif isinstance(item, dict):
                feat_name, params = next(iter(item.items()))
                if feat_name not in out:
                    out[feat_name] = []
                out[feat_name].append(params)
        return out

    def _add_features(self, new_df, col, vals, window_size, features):
        for feat_name, params in features.items():
            self._dispatch(new_df, col, vals, window_size, feat_name, params)

    def _dispatch(self, new_df, col, vals, window_size, feat_name, params):
        if feat_name == 'mean':
            new_df[f'{col}_mean'] = vals.mean(axis=1)

        elif feat_name == 'standard_deviation':
            new_df[f'{col}_standard_deviation'] = vals.std(axis=1)

        elif feat_name == 'minimum':
            new_df[f'{col}_minimum'] = vals.min(axis=1)

        elif feat_name == 'maximum':
            new_df[f'{col}_maximum'] = vals.max(axis=1)

        elif feat_name == 'range':
            new_df[f'{col}_range'] = vals.max(axis=1) - vals.min(axis=1)

        elif feat_name == 'slope':
            new_df[f'{col}_slope'] = self._simple_slope(vals, window_size)

        elif feat_name == 'skewness':
            from scipy.stats import skew
            new_df[f'{col}_skewness'] = skew(vals, axis=1)

        elif feat_name == 'kurtosis':
            from scipy.stats import kurtosis
            new_df[f'{col}_kurtosis'] = kurtosis(vals, axis=1)

        elif feat_name == 'quantile':
            for p in (params or []):
                q = p['q']
                label = str(q).rstrip('0').rstrip('.') if '.' in str(q) else str(q)
                new_df[f'{col}_quantile_{label}'] = np.quantile(vals, q, axis=1)

        elif feat_name == 'autocorrelation':
            for p in (params or []):
                f_agg = p['f_agg']
                maxlag = p['maxlag']
                result = self._autocorrelation(vals, f_agg, maxlag)
                new_df[f'{col}_autocorrelation_{f_agg}_{maxlag}'] = result

        elif feat_name == 'linear_trend':
            for p in (params or []):
                attr = p['attr']
                chunk_len = p['chunk_len']
                f_agg = p['f_agg']
                result = self._linear_trend(vals, window_size, attr, chunk_len, f_agg)
                new_df[f'{col}_linear_trend_{attr}_{chunk_len}_{f_agg}'] = result

        else:
            self._logger.warning("Unknown window feature '%s' - skipped", feat_name)

    @staticmethod
    def _simple_slope(vals, window_size):
        x = np.arange(window_size, dtype=float)
        x_dev = x - x.mean()
        ss_xx = (x_dev ** 2).sum()
        if ss_xx == 0:
            return np.zeros(len(vals))
        y_means = vals.mean(axis=1, keepdims=True)
        ss_xy = ((vals - y_means) * x_dev).sum(axis=1)
        return ss_xy / ss_xx

    @staticmethod
    def _autocorrelation(vals, f_agg, maxlag):
        """Compute autocorrelations at lags 1..maxlag then aggregate with f_agg."""
        means = vals.mean(axis=1, keepdims=True)
        centered = vals - means
        variance = (centered ** 2).sum(axis=1)  # (n,)

        ac_lags = []
        for lag in range(1, maxlag + 1):
            numerator = (centered[:, lag:] * centered[:, :vals.shape[1] - lag]).sum(axis=1)
            ac = np.where(variance > 0, numerator / variance, 0.0)
            ac_lags.append(ac)

        ac_matrix = np.stack(ac_lags, axis=1)  # (n, maxlag)

        if f_agg == 'mean':
            return ac_matrix.mean(axis=1)
        elif f_agg == 'median':
            return np.median(ac_matrix, axis=1)
        elif f_agg == 'var':
            return ac_matrix.var(axis=1)
        else:
            raise ValueError(f"Unsupported f_agg '{f_agg}' for autocorrelation")

    @staticmethod
    def _linear_trend(vals, window_size, attr, chunk_len, f_agg):
        """
        Split each window into non-overlapping chunks of chunk_len, fit a linear
        trend to each chunk, extract attr, then aggregate across chunks with f_agg.
        """
        n_chunks = window_size // chunk_len
        if n_chunks == 0:
            return np.zeros(len(vals))

        x = np.arange(chunk_len, dtype=float)
        x_mean = x.mean()
        x_dev = x - x_mean
        ss_xx = (x_dev ** 2).sum()

        attr_per_chunk = []
        for c in range(n_chunks):
            y = vals[:, c * chunk_len:(c + 1) * chunk_len]  # (n, chunk_len)
            y_mean = y.mean(axis=1)                          # (n,)
            y_centered = y - y_mean[:, None]

            if ss_xx == 0:
                slope = np.zeros(len(vals))
                intercept = y_mean
                rvalue = np.zeros(len(vals))
            else:
                ss_xy = (y_centered * x_dev).sum(axis=1)
                slope = ss_xy / ss_xx
                intercept = y_mean - slope * x_mean
                ss_yy = (y_centered ** 2).sum(axis=1)
                denom = np.sqrt(ss_xx * ss_yy)
                rvalue = np.where(denom > 0, ss_xy / denom, 0.0)

            if attr == 'slope':
                attr_per_chunk.append(slope)
            elif attr == 'intercept':
                attr_per_chunk.append(intercept)
            elif attr == 'rvalue':
                attr_per_chunk.append(rvalue)
            else:
                raise ValueError(f"Unsupported attr '{attr}' for linear_trend")

        attr_matrix = np.stack(attr_per_chunk, axis=1)  # (n, n_chunks)

        if f_agg == 'mean':
            return attr_matrix.mean(axis=1)
        elif f_agg == 'median':
            return np.median(attr_matrix, axis=1)
        elif f_agg == 'var':
            return attr_matrix.var(axis=1)
        else:
            raise ValueError(f"Unsupported f_agg '{f_agg}' for linear_trend")
