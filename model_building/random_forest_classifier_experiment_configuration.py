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
import copy

import numpy as np
import sklearn.ensemble as rf

import model_building.experiment_configuration as ec


class RandomForestClassifierExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for random forest classification

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the random forest model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values

    repair_hyperparameters()
        Repair and return hyperparameter values which cause the regressor to raise errors
    """
    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dict of str: object
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the classification problem to be solved

        prefix: list of str
            The prefix to be added to the signature of this experiment configuration
        """
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.RF_CLASSIFIER

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Returns
        -------
            The signature of the experiment
        """
        signature = prefix.copy()
        signature.append("n_estimators_" + str(self._hyperparameters['n_estimators']))
        signature.append("criterion_" + str(self._hyperparameters['criterion']))
        signature.append("max_depth_" + str(self._hyperparameters['max_depth']))
        signature.append("class_weight_" + str(self._hyperparameters['class_weight']))

        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor.fit(xdata, np.ravel(ydata))

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments
        """
        if not getattr(self, '_hyperparameters', None):
            self._regressor = rf.RandomForestClassifier()
        else:
            self._regressor = rf.RandomForestClassifier(
                n_estimators=self._hyperparameters['n_estimators'],
                criterion=self._hyperparameters['criterion'],
                max_depth=self._hyperparameters['max_depth'],
                class_weight=self._hyperparameters['class_weight'])

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'n_estimators': 100,
                'criterion': 'gini',
                'max_depth': None,
                'class_weight': None}

    def repair_hyperparameters(self, hypers):
        """
        Repair and return hyperparameter values which cause the regressor to raise errors

        Parameters
        ----------
        hypers: dict of str: object
            the hyperparameters to be repaired

        Return
        ------
        dict of str: object
            the repaired hyperparameters
        """
        new_hypers = copy.deepcopy(hypers)
        new_hypers['n_estimators'] = int(new_hypers['n_estimators'])
        if new_hypers['max_depth'] is not None:
            new_hypers['max_depth'] = int(new_hypers['max_depth'])
        return new_hypers
