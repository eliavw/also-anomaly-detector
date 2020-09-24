from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import numpy as np

VERBOSE = True


class AlsoAnomalyDetector:
    regressors = dict(dt=DecisionTreeRegressor, rf=RandomForestRegressor)

    def __init__(
        self,
        regressor="dt",
        n_folds=5,
        verbose=VERBOSE,
        random_state=42,
        contamination=0.1,
        **algorithm_kwargs
    ):
        # General params
        self.verbose = verbose

        # Metadata
        self.n_instances = None
        self.n_attributes = None
        self._attr_ids = None

        # Models
        self._models = None
        self._desc_ids = None
        self._targ_ids = None

        self.regressor_algorithm = regressor
        self.regressor_config = algorithm_kwargs

        # KFolds
        self.n_folds = n_folds
        self.kfold = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=random_state
        )
        self.folds = None

        # Residuals/Anomaly Scores
        self._residuals = None
        self._rrses = None
        self._scores = None
        self.contamination = contamination
        self._labels = None

        return

    def fit(self, X):
        # Set Metadata (you can only do this when you see the dataset)
        self.n_instances, self.n_attributes = X.shape
        self.attr_ids = self.n_attributes

        # Fit Models
        self._init_folds(X)
        self._init_desc_and_targ_ids()
        self._init_models(X)
        X_pred = self._fit_predict_models(X)

        # Get Scores
        self._labels = self._init_labels()
        self._residuals = self._get_residuals(X, X_pred)
        self._rrses = self._get_root_relative_squared_errors(X)
        self._scores = self._get_scores()
        self._set_labels()

        return

    def predict(self):
        return self.labels

    # Fit - Helpers
    def _init_folds(self, X):
        self.folds = list(self.kfold.split(X))
        return

    def _init_desc_and_targ_ids(self):
        d = []
        t = []
        for a in self.attr_ids:
            d.append(self.attr_ids - {a})
            t.append({a})

        self._desc_ids = d
        self._targ_ids = t

        return

    def _init_models(self, X):
        self.models = [
            self.regressor_algorithm(**self.regressor_config) for a in self.attr_ids
        ]
        return

    def _init_labels(self):
        # Initialize everything normal
        return np.zeros(self.n_instances)

    def _fit_predict_models(self, X):
        X_pred = np.zeros_like(X)
        for m_idx in range(self.n_models):
            desc_ids = list(self.desc_ids[m_idx])
            targ_ids = list(self.targ_ids[m_idx])

            x = X[:, desc_ids]
            y = X[:, targ_ids]

            y_pred = cross_val_predict(self.models[m_idx], x, y, cv=self.folds)
            X_pred[:, targ_ids] = y_pred.reshape(-1, len(targ_ids))

        return X_pred

    def _get_residuals(self, X_true, X_pred):
        return X_true - X_pred

    def _get_root_relative_squared_errors(self, X_true):
        sum_squared_residuals = np.sum(np.power(self.residuals, 2), axis=0)
        sum_squared_distance_to_means = np.sum(
            np.power(X_true - np.mean(X_true, axis=0), 2), axis=0
        )
        return np.sqrt(sum_squared_residuals / sum_squared_distance_to_means)

    def _get_scores(self):
        init_scores = np.power(self.residuals, 2)
        return np.sqrt(
            np.sum(self.weights * init_scores, axis=1) / np.sum(self.weights)
        )

    def _set_labels(self):
        anomaly_idxs = np.argpartition(self.scores, -self.int_contamination)[
            -self.int_contamination:
        ]
        self._labels[anomaly_idxs] = 1
        return

    # ----------
    # PROPERTIES
    # ----------
    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        assert isinstance(value, list)
        assert (
            len(value) == self.n_attributes
        ), "The amount of models must equal the amount of attributes"

        self._models = value
        return

    @property
    def n_models(self):
        return len(self.models)

    @property
    def residuals(self):
        return self._residuals

    @property
    def root_relative_squared_errors(self):
        return self._rrses

    @property
    def rrses(self):
        return self.root_relative_squared_errors

    @property
    def weights(self):
        return np.ones(self.n_attributes) - np.minimum(1, self.rrses)

    @property
    def decision_scores_(self):
        return self.scores

    @property
    def scores(self):
        if self._scores is not None:
            return self._scores
        else:
            self._scores = self._get_scores()
            return self._scores

    @property
    def labels(self):
        if self._labels is not None:
            return self._labels
        else:
            self._set_labels()
            return self._labels

    @labels.setter
    def labels(self, value):
        assert isinstance(value, np.ndarray)
        assert value.shape[0] == self.n_instances
        self._labels == value
        return

    @property
    def n_anomalies(self):
        return np.sum(self.labels)

    @property
    def int_contamination(self):
        # Contamination in integer.
        return int(self.contamination * self.n_instances)

    @property
    def regressor_algorithm(self):
        return self._regressor_algorithm

    @regressor_algorithm.setter
    def regressor_algorithm(self, value):
        if isinstance(value, str):
            self._regressor_algorithm = self.regressors[value]
        else:
            self._regressor_algorithm = value
        return

    @property
    def desc_ids(self):
        return self._desc_ids

    @property
    def targ_ids(self):
        return self._targ_ids

    @property
    def attr_ids(self):
        return self._attr_ids

    @attr_ids.setter
    def attr_ids(self, n):
        self._attr_ids = set(range(n))
        return

    @property
    def numeric_ids(self):
        return self.attr_ids - self.nominal_ids
