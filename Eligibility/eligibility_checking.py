import pandas as pd

class PredictionEligibilityChecker:
    MIN_TRANSACTIONS = 20
    MIN_ACTIVITY_DAYS = 14

    def __init__(self, user_data: pd.DataFrame) -> None:
        """
        Initialize the eligibility checker.

        Parameters
        ----------
        user_data : pandas.DataFrame
            Raw transaction data of a single user.
        """
        if not isinstance(user_data, pd.DataFrame):
            raise ValueError(
                'User data must be a pandas DataFrame.'
            )

        self.user_data = user_data
        
    def is_eligible(self) -> bool:
        """
        Check whether the user has enough activity
        and transactions to build a prediction model.

        Returns
        -------
        bool
            True if the user satisfies all minimum
            eligibility requirements.
        """
        has_min_activity = self._has_minimum_active_days()
        has_min_transaction = self._has_minimum_transactions()

        return has_min_activity and has_min_transaction

    def _has_minimum_transactions(self) -> bool:
        """
        Check whether the user has the minimum
        required number of valid transactions.

        Duplicate rows and rows with missing values
        are removed before counting.

        Returns
        -------
        bool
            True if the transaction count is greater
            than or equal to the configured threshold.
        """
        df = self.user_data.copy()

        df = df.drop_duplicates()

        df = df.filter(
            items=[
                col
                for col in df.columns
                if col.lower() != 'description'
            ]
        )

        df = df.dropna()

        return len(df) >= self.MIN_TRANSACTIONS


    def _make_daily(self) -> pd.DataFrame:
        """
        Aggregate transactions into daily totals.

        Returns
        -------
        pandas.DataFrame
            Daily income and expense values grouped
            by transaction type.
        """
        df = self.user_data.copy()

        df['Date'] = pd.to_datetime(
            df['Date Time']
        ).dt.date

        daily = (
            df.groupby(
                ['Date', 'Transaction Kind']
            )['Amount']
            .sum()
            .unstack(fill_value=0)
        )

        return daily

    def _has_minimum_active_days(self) -> bool:
        """
        Check whether the user has enough active days.

        Returns
        -------
        bool
            True if the number of active days is
            greater than or equal to the configured
            threshold.
        """
        daily = self._make_daily()

        return len(daily) >= self.MIN_ACTIVITY_DAYS

    
    def get_failure_reason(self) -> str:
        """
        Return a human-readable explanation of why
        the user is not eligible for prediction.

        Returns
        -------
        str
            Failure reason in Persian.
        """
        has_min_activity = self._has_minimum_active_days()
        has_min_transaction = self._has_minimum_transactions()

        if (
            not has_min_activity and
            not has_min_transaction
        ):
            return (
                'تعداد تراکنش‌ها و تعداد روزهای فعالیت '
                'کمتر از حد مورد نیاز است.'
            )

        elif not has_min_transaction:
            return (
                'تعداد تراکنش‌های انجام شده '
                'کمتر از حد مورد نیاز است.'
            )

        elif not has_min_activity:
            return (
                'تعداد روزهای فعالیت '
                'کمتر از حد مورد نیاز است.'
            )

        return 'خطای غیر منتظره رخ داده است.'