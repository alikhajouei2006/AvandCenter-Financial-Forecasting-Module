from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler
)
from sklearn import set_config

from ml_engine.EDA import (
    add_date_features,
    add_lag_features,
    finalize_features
)

set_config(transform_output="pandas")


class MakePipelines:
    """
    Creates and manages the preprocessing pipeline used for
    feature engineering and scaling.
    """

    def __init__(self) -> None:
        self.steps: Optional[list] = None
        self.pipeline: Optional[Pipeline] = None

    def make_steps(self) -> None:
        """
        Create the preprocessing pipeline steps.
        """

        self.steps = [
            (
                'datetime_features',
                FunctionTransformer(
                    add_date_features,
                    validate=False
                )
            ),
            (
                'lag_features',
                FunctionTransformer(
                    add_lag_features,
                    validate=False
                )
            ),
            (
                'finalize',
                FunctionTransformer(
                    finalize_features,
                    validate=False
                )
            ),
            (
                'scale',
                StandardScaler()
            )
        ]

    def make_pipeline(self) -> Pipeline:
        """
        Build and return a scikit-learn Pipeline object.

        Returns:
            Pipeline: Configured preprocessing pipeline.
        """

        if self.steps is None:
            self.make_steps()

        return Pipeline(self.steps)

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the preprocessing pipeline on the provided dataset.

        Args:
            X (pd.DataFrame):
                Raw feature dataframe.
        """

        pipeline = self.make_pipeline()
        pipeline.fit(X)
        self.pipeline = pipeline

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline and transform the dataset.

        Args:
            X (pd.DataFrame):
                Raw feature dataframe.

        Returns:
            pd.DataFrame:
                Preprocessed feature dataframe.
        """
        pipeline = self.make_pipeline()

        X_processed = pipeline.fit_transform(X)

        if not isinstance(
            X_processed,
            pd.DataFrame
        ):
            X_processed = pd.DataFrame(
                X_processed
            )

        self.pipeline = pipeline
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the already-fitted pipeline.

        Args:
            X (pd.DataFrame):
                Raw feature dataframe.

        Returns:
            pd.DataFrame:
                Preprocessed feature dataframe.

        Raises:
            ValueError:
                If the pipeline has not been fitted yet.
        """

        if self.pipeline is None:
            raise ValueError(   
                "Pipeline must be fitted before calling transform()."
            )
        
        X_processed = self.pipeline.transform(X)
        if not isinstance(
            X_processed,
            pd.DataFrame
        ):
            
            X_processed = pd.DataFrame(
                X_processed
            )

        return X_processed