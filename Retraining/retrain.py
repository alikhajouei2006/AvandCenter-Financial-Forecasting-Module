import pandas as pd

from Training.model_opt import ModelHyperparameterTuning


class RetrainManager:
    """
    Determines whether a model should be retrained
    and executes retraining when required.
    """

    MIN_NEW_TRANSACTION_COUNT = 10
    MIN_DAYS_AFTER_LAST_TRAIN = 7

    def __init__(
        self,
        user_log: pd.DataFrame,
        user_data: pd.DataFrame
    ) -> None:
        self.user_log = user_log
        self.user_data = user_data

    def has_new_transactions(self) -> bool:
        """
        Check whether enough new transactions
        have been added since the last training.

        Returns
        -------
        bool
            True if minimum transaction threshold
            has been reached.
        """
        current_size = len(self.user_data)

        trained_size = int(
            self.user_log[
                "training_data_size"
            ].iloc[0]
        )

        return (
            current_size - trained_size
            >= self.MIN_NEW_TRANSACTION_COUNT
        )

    def has_min_days_threshold(self) -> bool:
        """
        Check whether enough time has passed
        since the previous training run.

        Returns
        -------
        bool
            True if retraining interval
            has been satisfied.
        """
        last_trained = pd.to_datetime(
            self.user_log[
                "last_trained_at"
            ].iloc[0]
        )

        today = pd.Timestamp.now()

        return (
            (today - last_trained).days
            >= self.MIN_DAYS_AFTER_LAST_TRAIN
        )

    def should_retrain(self) -> bool:
        """
        Determine whether retraining should occur.

        Returns
        -------
        bool
            True if both transaction and time
            thresholds are satisfied.
        """
        return (
            self.has_new_transactions()
            and
            self.has_min_days_threshold()
        )

    def execute_retrain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimizer: ModelHyperparameterTuning
    ):
        """
        Execute model retraining if conditions
        are satisfied.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.

        y : pd.Series
            Training targets.

        optimizer : ModelHyperparameterTuning
            Hyperparameter optimization service.

        Returns
        -------
        object | None
            Best trained model if retraining
            occurred, otherwise None.
        """
        if not self.should_retrain():
            return None

        return optimizer.optimize(X, y)