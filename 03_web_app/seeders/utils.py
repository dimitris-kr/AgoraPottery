import pandas as pd
from sklearn.model_selection import train_test_split


def print_status(table_name, counter):
    status = "✅" if counter > 0 else "❎"
    adding = f"adding {counter}" if counter > 0 else "no additions"
    print(f"{status} {table_name}: {adding}")

def load_data(path_data):
    df = pd.read_csv(
        path_data,
        dtype={
            "Id": "string",
            "FullText": "string",
            "ImageFilename": "string",
        }
    )

    required_cols = {
        "Id", "FullText", "ImageFilename",
        "StartYear", "EndYear", "MidpointYear",
        "YearRange", "HistoricalPeriod", "ValidChronology"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # for col in required_cols:
    #     df[col] = df[col].replace("", None)
    #     df[col] = df[col].where(
    #         df[col].notna(), None
    #     )

    df = df[df["ValidChronology"] == True].copy()
    df.reset_index(drop=True, inplace=True)

    return df

def split_dataset(df, train_ratio, val_ratio, random_state):
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=random_state,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio := val_ratio / (1 - train_ratio)),
        random_state=random_state,
        shuffle=True
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df]).reset_index(drop=True)