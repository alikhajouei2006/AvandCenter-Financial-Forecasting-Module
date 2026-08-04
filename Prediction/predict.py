import datetime
from typing import List, Tuple

import pandas as pd
from sklearn.base import RegressorMixin

from ml_engine.pipelines import MakePipelines

class ForecastEngine:
    """
    Recursive multi-step forecasting engine.
    """

    def __init__(
        self,
        model: RegressorMixin,
        pipeline: MakePipelines
    ) -> None:
        
        self.model = model
        self.pipeline = pipeline


    def _add_future_row(
        self,
        df: pd.DataFrame,
        n_days: int
    ) -> pd.DataFrame:
        """
        Appends an empty future row.

        Parameters
        ----------
        df : pd.DataFrame
            Historical dataset.

        n_days : int
            Number of days ahead.

        Returns
        -------
        pd.DataFrame
            Dataset with future row appended.
        """
        future_date = (
            df["Date"].max()
            + datetime.timedelta(days=n_days)
        )

        new_row = pd.DataFrame(
            {
                "Date": [future_date],
                "NetBalance": [pd.NA],
            }
        )

        return pd.concat(
            [df, new_row],
            ignore_index=True
        )
    
    def predict_step(
        self,
        X_processed: pd.DataFrame
    ) -> float:
        """
        Predicts a single future step.

        Parameters
        ----------
        X_processed : pd.DataFrame
            Processed feature vector.

        Returns
        -------
        float
            Predicted net balance.
        """
        return float(
            self.model.predict(X_processed)[0]
        )

    def run(
        self,
        X: pd.DataFrame,
        n_days: int = 7
    ) -> Tuple[float, List[float]]:
        """
        Executes recursive forecasting.

        Parameters
        ----------
        X : pd.DataFrame
            Historical daily transaction data.

        n_days : int, default=7
            Forecast horizon.

        Returns
        -------
        Tuple[float, List[float]]
            Total predicted balance and
            daily predictions.
        """
        raw = X.copy()

        raw = (
            raw
            .sort_values("Date")
            .reset_index(drop=True)
        )

        predictions = []

        total_balance = float(raw["NetBalance"].sum())

        for step in range(1, n_days + 1):
            

            raw = self._add_future_row(raw, step)

            processed = self.pipeline.transform(raw)

            X_future = processed.iloc[[-1]]

            y_pred = self.predict_step(X_future)

            predictions.append(y_pred)

            raw.loc[
                raw.index[-1],
                "NetBalance"
            ] = y_pred

            total_balance += y_pred
        return total_balance, predictions