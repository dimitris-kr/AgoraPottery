"""
test_modal_spawn.py
-------------------
Smoke test for the deployed Modal `run_training` function.

Spawns the deployed function with a real-ish payload, polls for status until
the job terminates, and prints the outcome. The whole point: catch deployment
bugs (wrong app/function name, missing secrets, image build failures, runtime
errors inside run_training) BEFORE involving Render + ngrok in the loop.

Differences from a production trigger:
  - Bypasses /models/retrain/trigger entirely — calls `_spawn_modal_job`
    directly, the exact same helper trigger_retrain uses.
  - Uses items already in the current TrainingRun's splits (via load_items),
    so we don't create a new TrainingRun in the DB.
  - new_version="v_test" so uploads land in a throwaway folder.
  - webhook_url=None — Modal does the training + uploads but skips the
    callback, so this script doesn't touch DB promotion at all.

Pre-reqs:
  - `modal deploy Modal/modal_app.py` has run successfully.
  - `modal app list` shows agora-pottery-retrain.
  - Local CLI is authenticated (~/.modal.toml exists).

Usage (from 03_web_app/):
    python -m scripts.retraining_local_tests.test_modal_spawn
"""

import os
import time

from dotenv import load_dotenv

from database import SessionLocal, engine, Base
from services import build_model_configs, get_job_status
# _spawn_modal_job is intentionally underscored (internal helper) but we
# import it here on purpose — the test's whole job is to exercise that exact
# helper, the one trigger_retrain will use in production.
from services.retrain_service import _spawn_modal_job

# Sibling-module import — relies on PyCharm / runner adding the script dir to
# sys.path, same pattern as test_train_all.py.
from test_train_single import load_items


load_dotenv()

HF_IMAGE_REPO = os.getenv("HF_IMAGE_REPO")
HF_TFIDF_REPO = os.getenv("HF_TFIDF_REPO")
HF_VIT_REPO   = os.getenv("HF_VIT_REPO")

if not (HF_IMAGE_REPO and HF_TFIDF_REPO and HF_VIT_REPO):
    raise EnvironmentError("HF_IMAGE_REPO, HF_TFIDF_REPO, HF_VIT_REPO must be set in .env")

NEW_VERSION   = "v_test"
PREV_VERSION  = "v1"
POLL_INTERVAL = 10  # seconds between status checks


def main():
    Base.metadata.create_all(bind=engine)

    print("\n** Loading items + model configs from DB **")
    items = load_items()                # {"train": [...], "val": [...]}
    db = SessionLocal()
    try:
        model_configs = build_model_configs(db)
    finally:
        db.close()
    print(f"   {len(items['train'])} train / {len(items['val'])} val items")
    print(f"   {len(model_configs)} model variants")

    payload = {
        "training_run_id": -1,                    # sentinel; webhook is None so it's never read
        "new_version":     NEW_VERSION,
        "prev_version":    PREV_VERSION,
        "items_train":     items["train"],
        "items_val":       items["val"],
        "hf_tfidf_repo":   HF_TFIDF_REPO,
        "hf_vit_repo":     HF_VIT_REPO,
        "hf_images_repo":  HF_IMAGE_REPO,
        "models":          model_configs,
        "webhook_url":     None,                  # Modal trains + uploads but won't call back
    }

    print(f"\n** Spawning Modal job (new_version={NEW_VERSION!r}) **")
    job_id = _spawn_modal_job(payload)
    print(f"   job_id={job_id}")

    print(f"\n** Polling every {POLL_INTERVAL}s **")
    elapsed = 0
    while True:
        status = get_job_status(job_id)
        print(f"   [{elapsed:>4}s] status={status.status}")
        if status.status in ("success", "failure", "not_found"):
            break
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    print("\n")
    if status.status == "success":
        print("   ✓ run_training returned cleanly.")
        if status.result:
            print(f"   Returned dict keys: {list(status.result.keys())}")
            # run_training returns {"status", "results", "error"}
            inner_status  = status.result.get("status")
            inner_results = status.result.get("results", [])
            inner_error   = status.result.get("error")
            print(f"   Inner status:  {inner_status}")
            print(f"   Inner results: {len(inner_results)} entries")
            for result in inner_results: print(f"      {result}")
            if inner_error:
                print(f"   Inner error:   {inner_error[:500]}...")
    elif status.status == "failure":
        print("   ✗ Modal raised error inside run_training:")
        print(f"   {status.error}")
    else:
        print(f"   ? Unexpected terminal status: {status.status}")

    print(f"\n   v_test/ artifacts may now exist on HF. Clean up via either:")
    print(f"     - HF web UI (manual)")
    print(f"     - python -m scripts.retraining_local_tests.test_finalize error")


if __name__ == "__main__":
    main()
