"""
Copyright 2019 Marco Lattuada
Copyright 2025 Federica Filippini
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_pinball_loss
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(np.average(mape, axis=0), axis = 0)


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def to_labels(y):
    return np.asarray(y).ravel().astype(int)


def _average_for(y_true, y_pred):
    # binary or multiclass output
    n_classes = np.unique(np.concatenate([y_true, y_pred])).size
    return "binary" if n_classes <= 2 else "macro"


def accuracy(y_true, y_pred):
    return accuracy_score(to_labels(y_true), to_labels(y_pred))


def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(to_labels(y_true), to_labels(y_pred))


def f1(y_true, y_pred):
    yt, yp = to_labels(y_true), to_labels(y_pred)
    return f1_score(yt, yp, average=_average_for(yt, yp), zero_division=0)


def precision(y_true, y_pred):
    yt, yp = to_labels(y_true), to_labels(y_pred)
    return precision_score(yt, yp, average=_average_for(yt, yp), zero_division=0)


def recall(y_true, y_pred):
    yt, yp = to_labels(y_true), to_labels(y_pred)
    return recall_score(yt, yp, average=_average_for(yt, yp), zero_division=0)


class Metrics:
    def __init__(self, quantile: float = 0.5, task: str = 'regression'):
        regression_metrics = {
            "MAPE": {
              "func": mean_absolute_percentage_error,
              "attributes": {},
              "comp": (lambda x,y : x < y)  # lower is better
            }, 
            "RMSE": {
              "func": root_mean_squared_error,
              "attributes": {},
              "comp": (lambda x,y : x < y)  # lower is better
            },
            "R^2": {
              "func": r2_score,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            }, 
            "MAE": {
              "func": mean_absolute_error,
              "attributes": {},
              "comp": (lambda x,y : x < y)  # lower is better
            }, 
            "MSE": {
              "func": mean_squared_error,
              "attributes": {},
              "comp": (lambda x,y : x < y)  # lower is better
            },
            "QL": {
              "func": mean_pinball_loss,
              "attributes": {"alpha": quantile},
              "comp": (lambda x,y : x < y)  # lower is better
            }
        }
        classification_metrics = {
            "Accuracy": {
              "func": accuracy,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            },
            "F1": {
              "func": f1,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            },
            "Precision": {
              "func": precision,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            },
            "Recall": {
              "func": recall,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            },
            "BalancedAccuracy": {
              "func": balanced_accuracy,
              "attributes": {},
              "comp": (lambda x,y : x > y)  # greater is better
            }
        }
        self.task = task
        self._metrics_dict = classification_metrics if task == "classification" else regression_metrics

    def compute_metric(self, metric, real_y, predicted_y):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["func"](real_y, predicted_y, **self._metrics_dict[metric]["attributes"])
        else:
            return None
    
    def compute_metrics(self, real_y, predicted_y):
        metrics = {}
        for metric in self._metrics_dict:
            metrics[metric] = self.compute_metric(metric, real_y, predicted_y)
        return metrics
    
    def greater_is_better(self, metric):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["comp"](3,2)
        else:
            return None
    
    def get_metric_operator(self, metric):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["func"]
        else:
            return None
    
    def get_comparison_operator(self, metric):
        if metric in self._metrics_dict:
            return self._metrics_dict[metric]["comp"]
        else:
            return None
    
    def supported_metrics(self):
        return list(self._metrics_dict.keys())
