import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from the Date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a Date column.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with date-related features.
    """
    df = df.copy()

    df = df.reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    df["DayOfWeek"] = df["Date"].dt.dayofweek.astype(float)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(float)

    df["DayOfMonth"] = df["Date"].dt.day.astype(float)
    df["Month"] = df["Date"].dt.month.astype(float)

    df["WeekOfMonth"] = (
        ((df["Date"].dt.day - 1) // 7) + 1
    ).astype(float)

    df = df.sort_values("Date").reset_index(drop=True)

    df.drop(columns=["Date"], inplace=True)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "NetBalance" not in df.columns:
            raise ValueError(
                "Input must contain NetBalance column."
            )

    shifted = df["NetBalance"].shift(1)

    df["NetBalance_lag1"] = shifted

    df["NetBalance_lag7"] = (
        df["NetBalance"].shift(7)
    )   

    df["NetBalance_rollmean7"] = (
        shifted
        .rolling(7, min_periods=7)
        .mean()
    )

    df["NetBalance_rollstd7"] = (
        shifted
        .rolling(7, min_periods=7)
        .std()
    )

    return df

def finalize_features(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Remove unused columns and handle missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for modeling.
    """
    df = df.copy()

    drop_cols = [
        col
        for col in [
            "NetBalance",
            "Date"
        ]
        if col in df.columns
    ]

    df.drop(columns=drop_cols, inplace=True)

    df.fillna(0, inplace=True)

    return df