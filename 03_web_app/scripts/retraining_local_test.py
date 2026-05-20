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
    extract_vit_features,
    get_regression_targets,
    refit_y_scaler,
    scale_y,
    get_classification_targets,
    refit_y_encoder,
    encode_y,
    download_config, train_single_model,
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

def print_info_targets(y):
    print("{")
    for subset in y.keys():
        indent = "\t"
        print(f"{indent}{subset}: ", end="")

        y_type = type(y[subset])
        print()
        indent = 2 * "\t"
        print(f"{indent}{y_type}")
        print(f"{indent}shape   = {y[subset].shape}")

    print("}")

def extract_features(items):
    print("\n** Step 1: Extracting features **")
    vectorizer = refit_tfidf_vectorizer(items["train"])
    X = {
        subset: {
            "tfidf": extract_tfidf_features(items[subset], vectorizer),
            "vit": extract_vit_features(items[subset], HF_IMAGE_REPO, HF_TOKEN)
        }
        for subset in SUBSETS
    }
    print_info_features(X)
    return X


def prepare_targets(items):
    print("\n** Step 2: Scaling targets **")

    y = {}

    y["regression"] = {subset: get_regression_targets(items[subset]) for subset in SUBSETS}
    y_scaler = refit_y_scaler(y["regression"]["train"])
    y["regression"] = {subset: scale_y(y["regression"][subset], y_scaler) for subset in y["regression"].keys()}

    y["classification"] = {subset: get_classification_targets(items[subset]) for subset in SUBSETS}
    y_encoder = refit_y_encoder(y["classification"]["train"])
    y["classification"] = {subset: encode_y(y["classification"][subset], y_encoder) for subset in y["classification"].keys()}


    for task in y.keys():
        print(task)
        print_info_targets(y[task])
        print()

    return y, y_scaler, y_encoder

def test_train(X, y, y_scaler, y_encoder, feature_keys, task, config_repo):
    print("\n** Step 3: Training **")

    config = download_config(config_repo, PREV_VERSION, HF_TOKEN)
    print(f"  Config: {config}")

    _X = {subset: {ft: X[subset][ft]} for subset in X.keys() for ft in X[subset].keys() if ft in feature_keys}
    model, metadata, updated_config = train_single_model(
        _X,
        y[task],
        task,
        config,
        y_encoder=y_encoder
    )

    print(f"\n  ✓ Training complete")
    print(f"  val_loss : {metadata['val_loss']:.4f}")
    print(f"  scores   : {metadata['scores']}")
    print(f"  time     : {metadata['time']:.1f}s")

if __name__ == "__main__":
    items = load_items()
    X = extract_features(items)
    y, y_scaler, y_encoder = prepare_targets(items)

    test_train(X, y, y_scaler, y_encoder, feature_keys = ["tfidf"], task="classification", config_repo=CLF_TFIDF_REPO)

