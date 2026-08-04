from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit
)

import pandas as pd


class ModelHyperparameterTuning:
    """
    Performs hyperparameter optimization for
    RandomForestRegressor using RandomizedSearchCV.
    """

    def __init__(self) -> None:
        self.model = RandomForestRegressor(
            random_state=42
        )

        self.params = self._create_params()

    def _create_params(self) -> dict:
        """
        Create hyperparameter search space.

        Returns
        -------
        dict
            Parameter grid used by RandomizedSearchCV.
        """
        return  {
            "n_estimators": [100, 200, 300],

            "max_depth": [3, 5, 7, None],

            "min_samples_split": [2, 5, 10],

            "min_samples_leaf": [1, 2, 5],

            "max_features": [
                "sqrt",
                0.5,
                None
            ]
        }

    def _create_search(
        self
    ) -> RandomizedSearchCV:
        """
        Create RandomizedSearchCV object.

        Returns
        -------
        RandomizedSearchCV
            Configured search object.
        """
        tscv = TimeSeriesSplit(
            n_splits=5
        )

        return RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.params,
            n_iter=25,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42,
            
        )

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> RegressorMixin:
        """
        Find the best model using
        randomized hyperparameter search.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.

        y : pd.Series
            Target values.

        Returns
        -------
        RegressorMixin
            Best estimator found during search.
        """
        search = self._create_search()

        search.fit(X, y)

        return search.best_estimator_