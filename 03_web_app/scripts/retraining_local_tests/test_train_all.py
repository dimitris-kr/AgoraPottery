"""
test_train_all.py
--------------------------
End-to-end local test of the Modal `run_training` orchestrator.

Loads items + model configs from the DB, builds the payload, then invokes
run_training from Modal.retrain so the function runs locally.
Uploads go to HF under `new_version="v_test"`.

"""

import os

from dotenv import load_dotenv

from database import SessionLocal, engine, Base
from models import Model
# Import the plain orchestration function — no Modal runtime involved.
# The decorated cloud version lives in Modal.modal_app (not imported here).
from Modal.retrain import run_training
from services import build_model_configs

# Reuse the item-loading helper from the original local-test script
from test_train_single import load_items

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_IMAGE_REPO = os.getenv("HF_IMAGE_REPO")
HF_TFIDF_REPO = os.getenv("HF_TFIDF_REPO")
HF_VIT_REPO = os.getenv("HF_VIT_REPO")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set in .env")
if not HF_IMAGE_REPO or not HF_TFIDF_REPO or not HF_VIT_REPO:
    raise EnvironmentError("HF_IMAGE_REPO, HF_TFIDF_REPO and HF_VIT_REPO must be set in .env")

# Throwaway version — uploaded artifacts under "v_test/"
NEW_VERSION = "v_test"
PREV_VERSION = "v1"

# Skip webhook for local testing (no FastAPI listener running)
WEBHOOK_URL = None


# ─────────────────────────────────────────────
# STEP 1 — Build the list of model configs from the DB
# ─────────────────────────────────────────────

def load_model_configs() -> list[dict]:
    """
    One dict per Model row:
        {"model_id": int, "model_repo": hf_repo_id,
         "task": "classification"|"regression",
         "feature_keys": ["tfidf"] | ["vit"] | ["tfidf", "vit"]}

    feature_keys are sorted alphabetically — must match the canonical order
    train_single_model uses (and predict-time's alphabetical fallback).
    """
    print("\n** Loading model configs from DB **")
    db = SessionLocal()
    try:
        configs = build_model_configs(db)
    finally:
        db.close()

    for c in configs:
        print(f"   id={c['model_id']:<3} {c['task']:<14} | {"+".join(c['feature_keys']):<9} → {c['model_repo']}")

    return configs


# ─────────────────────────────────────────────
# STEP 2 — Run
# ─────────────────────────────────────────────

def main():
    Base.metadata.create_all(bind=engine)

    items = load_items()  # {"train": [...], "val": [...]}
    model_configs = load_model_configs()

    payload = {
        "training_run_id": -1,  # local test — not a real DB run
        "new_version": NEW_VERSION,
        "prev_version": PREV_VERSION,
        "items_train": items["train"],
        "items_val": items["val"],
        "hf_tfidf_repo": HF_TFIDF_REPO,
        "hf_vit_repo": HF_VIT_REPO,
        "hf_images_repo": HF_IMAGE_REPO,
        "models": model_configs,
        "webhook_url": WEBHOOK_URL,
    }

    print(f"\n** Running run_training with new_version={NEW_VERSION!r} **")
    print(f"   {len(payload['items_train'])} train / {len(payload['items_val'])} val items")
    print(f"   {len(model_configs)} model variants")

    # Plain Python call — no Modal runtime. HF_TOKEN comes from the local .env
    # (Modal secrets aren't available here).
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    result = run_training(payload)

    print(f"\n** Result: status={result['status']} **")
    # get task / feature_keys by model_id
    label_by_id = {
        c["model_id"]: f"{c['task']:<14} | {'+'.join(c['feature_keys']):<9}"
        for c in model_configs
    }
    for r in result["results"]:
        print(f"   id={r['model_id']:<3} {label_by_id.get(r['model_id'], '?')}  val_loss={r['val_loss']:.4f}")
    if result["error"]:
        print(f"\nERROR:\n{result['error']}")


if __name__ == "__main__":
    main()
