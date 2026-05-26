"""
test_finalize.py
------------------------
Standalone tester for services.retrain_service.finalize_retrain.

Bypasses HTTP/Modal entirely — calls the service function directly with a
hand-crafted WebhookPayloadSchema. Useful to verify the success and error
paths (including HF folder cleanup) before deploying on Modal.

Notes:
  - scripts/retraining_local_tests/test_train_all.py must have run
    at least once, so v_test/ artifacts exist on every HF repo.
  - 'success' promotes those artifacts in the DB without re-uploading anything.
  - 'error' triggers the rollback and best-effort HF cleanup of v_test/ folders.
  - Test TrainingRuns are marked with split_strategy="test-finalize" so the
    cleanup command can identify and remove them safely.
"""

from dotenv import load_dotenv

from database import SessionLocal, engine, Base
from models import (
    Model, ModelVersion, TrainingRun, FeatureSet,
    PotteryItem, PotteryItemInTrainingRun, ChronologyLabel,
)
from schemas import WebhookPayloadSchema, ModelTrainingResultSchema
from services import finalize_retrain

load_dotenv()


NEW_VERSION         = "v_test"
PREV_VERSION        = "v1"
TEST_SPLIT_MARKER   = "test-finalize"   # marks TrainingRuns created by this script


# ─────────────────────────────────────────────
# SETUP — mimic the pre-spawn state trigger_retrain leaves the DB in
# ─────────────────────────────────────────────

def _create_fake_training_run(db) -> TrainingRun:
    """
    Mimics what trigger_retrain does before spawning Modal:
      - Flip all current TrainingRuns to is_current=False
      - Create a new TrainingRun with is_current=False
      - Seed a few PotteryItemInTrainingRun rows with new TrainingRun
    """
    db.query(TrainingRun).filter(TrainingRun.is_current == True).update(
        {TrainingRun.is_current: False}
    )
    new_run = TrainingRun(
        split_strategy=TEST_SPLIT_MARKER,
        random_state=42,
        is_current=False,
    )
    db.add(new_run)
    db.flush()

    pottery_items = (
        db.query(PotteryItem)
        .join(PotteryItem.chronology_label)
        .join(ChronologyLabel.historical_period)
        .limit(5)
        .all()
    )
    for i, item in enumerate(pottery_items):
        db.add(PotteryItemInTrainingRun(
            training_run_id=new_run.id,
            pottery_item_id=item.id,
            split="train" if i < 4 else "val",
        ))

    db.commit()
    db.refresh(new_run)
    return new_run


def _build_fake_results(db) -> list[ModelTrainingResultSchema]:
    """One ModelTrainingResultSchema per Model row, with task-appropriate fake scores."""
    models = db.query(Model).all()
    results: list[ModelTrainingResultSchema] = []
    for m in models:
        if m.task.name.lower() == "classification":
            scores = {
                "accuracy":  [0.72],
                "precision": [0.68],
                "recall":    [0.65],
                "f1":        [0.66],
            }
        else:
            scores = {
                "mae":   [42.0, 18.5],
                "rmse":  [56.0, 24.0],
                "r2":    [0.55, 0.42],
                "medae": [38.0, 16.0],
            }
        results.append(ModelTrainingResultSchema(
            model_id=m.id,
            val_loss=0.5,
            train_time=12.3,
            scores=scores,
            train_sample_size=5,
        ))
    return results


# ─────────────────────────────────────────────
# MODE: success
# ─────────────────────────────────────────────

def test_success():
    db = SessionLocal()
    try:
        print("\n** Setup **")
        new_run = _create_fake_training_run(db)
        print(f"   Created TrainingRun id={new_run.id} (marker={TEST_SPLIT_MARKER!r}, is_current={new_run.is_current})")

        results = _build_fake_results(db)
        payload = WebhookPayloadSchema(
            training_run_id=new_run.id,
            new_version=NEW_VERSION,
            status="success",
            error=None,
            results=results,
        )
        print(f"   Built webhook payload: {len(results)} results, new_version={NEW_VERSION!r}")

        print("\n** Calling finalize_retrain(db, payload) **")
        result = finalize_retrain(db, payload)
        print(f"   → finalize_retrain returned: status={result.status}  models_updated={result.models_updated}")

        # Refresh + inspect
        db.expire_all()
        print("\n** DB state after **")

        new_mvs = (
            db.query(ModelVersion)
            .filter(ModelVersion.version == NEW_VERSION, ModelVersion.is_current == True)
            .all()
        )
        print(f"   ModelVersions (version={NEW_VERSION!r}, is_current=True): {len(new_mvs)} records")
        for mv in new_mvs:
            print(f"      id={mv.id} model_id={mv.model_id} "
                  f"val_loss={mv.val_loss} val_acc={mv.val_accuracy} val_mae={mv.val_mae} "
                  f"train_time={mv.train_time}")

        still_current_old = (
            db.query(ModelVersion)
            .filter(ModelVersion.version != NEW_VERSION, ModelVersion.is_current == True)
            .count()
        )
        print(f"   Non-{NEW_VERSION} ModelVersions still is_current=True: {still_current_old} (0 expected)")

        current_run = db.query(TrainingRun).filter(TrainingRun.is_current == True).one_or_none()
        print(f"   Current TrainingRun: id={current_run.id if current_run else None} "
              f"({new_run.id} expected)")

        fs_versions = {fs.feature_type: fs.current_version for fs in db.query(FeatureSet).all()}
        print(f"   FeatureSet versions: {fs_versions} ({NEW_VERSION!r} expected for all)")
    finally:
        db.close()


# ─────────────────────────────────────────────
# MODE: error
# ─────────────────────────────────────────────

def test_error():
    db = SessionLocal()
    try:
        print("\n** Setup **")
        new_run = _create_fake_training_run(db)
        new_run_id = new_run.id

        prev_run = (
            db.query(TrainingRun)
            .filter(TrainingRun.id != new_run_id)
            .order_by(TrainingRun.id.desc())
            .first()
        )
        prev_run_id = prev_run.id if prev_run else None
        print(f"   Created TrainingRun id={new_run_id} (is_current=False)")
        print(f"   Previous TrainingRun id={prev_run_id} (will be restored after rollback)")

        payload = WebhookPayloadSchema(
            training_run_id=new_run_id,
            new_version=NEW_VERSION,
            status="error",
            error="Simulated training failure for testing",
            results=[],
        )

        print("\n** Calling finalize_retrain(db, payload) **")
        print("   (this will also attempt to delete v_test/ folders on HF — best-effort)")
        result = finalize_retrain(db, payload)
        print(f"   → finalize_retrain returned status={result.status}")

        db.expire_all()
        print("\n** DB state after **")

        gone = db.get(TrainingRun, new_run_id)
        print(f"   New TrainingRun id={new_run_id}: {'deleted ✓' if gone is None else 'STILL PRESENT ✗'}")

        if prev_run_id:
            restored = db.get(TrainingRun, prev_run_id)
            mark = "✓" if (restored and restored.is_current) else "✗ NOT"
            print(f"   Previous TrainingRun id={prev_run_id} is_current=True: {mark} restored")

        orphan_pivots = (
            db.query(PotteryItemInTrainingRun)
            .filter(PotteryItemInTrainingRun.training_run_id == new_run_id)
            .count()
        )
        print(f"   Orphan PotteryItemInTrainingRun rows: {orphan_pivots} (0 expected)")

        print("\n** No DB cleanup needed — error path already rolled back **")
    finally:
        db.close()


# ─────────────────────────────────────────────
# MODE: cleanup — undo a success test
# ─────────────────────────────────────────────

def cleanup():
    """Idempotent: safe to run repeatedly. Reverts what test_success() did."""
    db = SessionLocal()
    try:
        # 1. Delete every ModelVersion at v_test
        mv_count = db.query(ModelVersion).filter(ModelVersion.version == NEW_VERSION).delete()
        print(f"   Deleted {mv_count} ModelVersion(s) at version={NEW_VERSION!r}")

        # 2. Restore v1 ModelVersions to is_current=True
        prev_mvs = db.query(ModelVersion).filter(ModelVersion.version == PREV_VERSION).all()
        for mv in prev_mvs:
            mv.is_current = True
        print(f"   Set {len(prev_mvs)} {PREV_VERSION!r} ModelVersion(s) to is_current=True")

        # 3. Reset FeatureSet.current_version → v1
        fs_count = db.query(FeatureSet).update({FeatureSet.current_version: PREV_VERSION})
        print(f"   Reset {fs_count} FeatureSet(s) to current_version={PREV_VERSION!r}")

        # 4. Delete every test TrainingRun (marker on split_strategy) + their pivot rows
        test_runs = (
            db.query(TrainingRun)
            .filter(TrainingRun.split_strategy == TEST_SPLIT_MARKER)
            .all()
        )
        for tr in test_runs:
            pitr_count = db.query(PotteryItemInTrainingRun).filter(
                PotteryItemInTrainingRun.training_run_id == tr.id
            ).delete()
            print(f"   Deleted {pitr_count} test PotteryItemInTrainingRun rows with training_run_id=={tr.id}")
            db.delete(tr)
        print(f"   Deleted {len(test_runs)} test TrainingRun(s)")

        # 5. Restore the most-recent non-test TrainingRun to is_current=True
        live_run = (
            db.query(TrainingRun)
            .filter(TrainingRun.split_strategy != TEST_SPLIT_MARKER)
            .order_by(TrainingRun.id.desc())
            .first()
        )
        if live_run:
            live_run.is_current = True
            print(f"   Set TrainingRun id={live_run.id} to is_current=True")

        db.commit()
        print("\n   ✓ Cleanup complete")
    finally:
        db.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)

    mode = input("Choose Mode: success | error | cleanup\n")

    if mode == "success":
        test_success()
    elif mode == "error":
        test_error()
    elif mode == "cleanup":
        cleanup()
    else:
        print("Unknown mode. Try again.")
