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
import sklearn.linear_model as lr

import model_building.experiment_configuration as ec


class LogisticRegressionExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for logistic regression

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the logistic model

    print_model()
        Print the representation of the generated model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values
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
        assert prefix
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.LR_LOGISTIC

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
        assert isinstance(prefix, list)
        signature = prefix.copy()
        signature.append("C_" + str(self._hyperparameters['C']))
        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor.fit(xdata, np.ravel(ydata))
        self._logger.debug("Model built")
        for idx, col_name in enumerate(self.get_x_columns()):
            self._logger.debug("The coefficient for %s is %f", col_name, self._regressor.coef_[0][idx])

    def print_model(self):
        """
        Print the representation of the generated model
        """
        initial_string = "LogisticRegression coefficients (log-odds):\n"
        ret_string = initial_string
        coefficients = self._regressor.coef_[0]
        columns = self.get_x_columns()

        assert len(columns) == len(coefficients)

        # Show coefficients in order of decresing absolute value
        idxs = np.argsort(np.abs(coefficients))[::-1]
        signif_digits = 4
        for i in idxs:
            column = columns[i]
            coefficient = coefficients[i]
            ret_string += " + " if ret_string != initial_string else "   "
            coeff = str(round(coefficient, signif_digits))
            ret_string = ret_string + "(" + str(coeff) + " * " + column + ")\n"
        coeff = str(round(self._regressor.intercept_[0], signif_digits))
        ret_string = ret_string + " + (" + coeff + ")"
        return ret_string

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments

        max_iter is raised with respect to the sklearn default to avoid convergence warnings
        """
        if not getattr(self, '_hyperparameters', None):
            self._regressor = lr.LogisticRegression(max_iter=1000)
        else:
            self._regressor = lr.LogisticRegression(C=self._hyperparameters['C'], max_iter=1000)

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'C': 1.0}
