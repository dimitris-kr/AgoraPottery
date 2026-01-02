import pandas as pd
import json
from huggingface_hub import hf_hub_download
from sqlalchemy import and_, func


from models import TrainingRun, PotteryItemInTrainingRun


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

def get_current_training_run(db):
    tr = (
        db.query(TrainingRun)
        .filter(TrainingRun.is_current == True)
        .one_or_none()
    )
    if not tr:
        raise RuntimeError("No current TrainingRun found")
    return tr

def get_train_sample_size(db, training_run_id: int) -> int:
    return (
        db.query(func.count(PotteryItemInTrainingRun.pottery_item_id))
        .filter(
            PotteryItemInTrainingRun.training_run_id == training_run_id,
            PotteryItemInTrainingRun.split.in_(["train", "val"])
        )
        .scalar()
    )

def load_metadata_from_hf(repo_id: str, version: str = "v1") -> dict:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=f"{version}/metadata.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
