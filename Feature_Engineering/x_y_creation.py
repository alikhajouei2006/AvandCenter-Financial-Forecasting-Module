from typing import Optional

import pandas as pd


class XyCreation:
    """
    Creates feature matrix (X) and target vector (y)
    from raw transaction data.
    """

    def __init__(
        self,
        user_data: pd.DataFrame
    ) -> None:
        self.user_data = user_data

        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.daily = None

    def _make_daily(self) -> pd.DataFrame:
        """
        Aggregate transactions into daily
        income and expense values.

        Returns
        -------
        pd.DataFrame
            Daily aggregated transaction data.
        """
        df = self.user_data.copy()

        df["Date"] = pd.to_datetime(
            df["Date Time"]
        ).dt.date

        daily = (
            df.groupby(
                ["Date", "Transaction Kind"]
            )["Amount"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        return daily

    def create_x(self) -> pd.DataFrame:
        """
        Create feature matrix.

        Returns
        -------
        pd.DataFrame
            Daily transaction dataset.
        """
        self.daily = self._make_daily()
        self.X = self.daily

        self.X['NetBalance'] = self.X['Income'] - self.X['Expense']
        self.X = self.X.drop(columns=['Income', 'Expense'])

        return self.X

    def create_y(self) -> pd.Series:
        """
        Create target variable
        (daily net balance).

        Returns
        -------
        pd.Series
            Daily net balance.
        """
        if self.X is None:
            raise ValueError(
                "X must be created before y."
            )

        self.y = (
            self.daily["Income"]
            - self.daily["Expense"]
        )

        return self.y