import os

from dotenv import load_dotenv

from database import SessionLocal, engine, Base
from services import get_current_training_run
from models import (
    PotteryItem,
    ChronologyLabel,
    HistoricalPeriod,
    PotteryItemInTrainingRun,
    TrainingRun,
)
from Modal import (
    refit_tfidf_vectorizer,
    extract_tfidf_features,
    extract_vit_features
)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set in .env")

HF_IMAGE_REPO = os.getenv("HF_IMAGE_REPO")
HF_TFIDF_REPO = os.getenv("TFIDF_REPO")

PREV_VERSION = "v1"
CLF_TFIDF_REPO = "dimitriskr/agora_pottery_chronology_classifier_tfidf"

SUBSETS = ["train", "val"]

# ─────────────────────────────────────────────
# STEP 0 — Load items from DB
# ─────────────────────────────────────────────

def get_items_by_split(db, current_run, split: str, samples: int) -> list[dict]:
    limit = samples * 0.9 if split == "train" else samples * 0.1
    rows = (
        db.query(PotteryItem)
        .join(PotteryItem.chronology_label)
        .join(ChronologyLabel.historical_period)
        .join(PotteryItem.in_training_run)
        .filter(
            PotteryItemInTrainingRun.training_run_id == current_run.id,
            PotteryItemInTrainingRun.split == split,
        )
        .limit(limit)
        .all()
    )
    return [
        {
            "pottery_item_id": item.id,
            "description": item.description,
            "image_path": item.image_path,
            "historical_period": item.chronology_label.historical_period.name,
            "start_year": item.chronology_label.start_year,
            "year_range": item.chronology_label.year_range,
        }
        for item in rows
    ]


def load_items():
    print("\n** Step 0: Loading items from DB **")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    current_run = get_current_training_run(db)
    print(f"   Current training run: id={current_run.id}")

    items = {}
    for subset in SUBSETS:
        items[subset] = get_items_by_split(db, current_run, subset, 100)

    db.close()

    print(f"   Loaded:")
    for subset in SUBSETS:
        print(f"      {len(items[subset])} {subset} items")

    for subset in SUBSETS:
        print(f"   {subset} stats:")
        print(f"      periods: {set(it["historical_period"] for it in items[subset])}")
        print(f"      image items: {sum(1 for it in items[subset] if it['image_path'])}")

    return items

def print_info_features(X):
    print("{")
    for subset in X.keys():
        indent = "\t"
        print(f"{indent}{subset}: " + "{")
        for method in X[subset].keys():
            indent = 2 * "\t"
            print(f"{indent}{method}: ")
            indent = 3 * "\t"
            print(f"{indent}{type(X[subset][method])}")
            print(f"{indent}shape = {X[subset][method].shape}, ")
        print("\t},")
    print("}")

def extract_features(items):
    print("\n** Step 1: Extracting features **")

    X = {
        "tfidf": {},
        "vit": {}
    }

    vectorizer = refit_tfidf_vectorizer(items["train"])

    for subset in SUBSETS:
        X["tfidf"][subset] = extract_tfidf_features(items[subset], vectorizer)
        X["vit"][subset] = extract_vit_features(items[subset], HF_IMAGE_REPO, HF_TOKEN)

    print_info_features(X)

if __name__ == "__main__":
    items = load_items()
    extract_features(items)