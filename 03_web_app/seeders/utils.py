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

    df = df[df["ValidChronology"] == True].copy()
    df.reset_index(drop=True, inplace=True)

    return df

