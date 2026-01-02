from models import Task, Model, FeatureSet, ModelUsesFeatureSet, ModelVersion, Target, ModelHasTarget, TrainingRun, \
    PotteryItemInTrainingRun
import json
from huggingface_hub import hf_hub_download
from sqlalchemy import and_, func

from seeders.utils import load_metadata_from_hf, get_current_training_run, get_train_sample_size, print_status


def seed_models(db):
    MODELS = [
        # classification
        ("agora_pottery_chronology_classifier_tfidf", "Classification", ["tfidf"]),
        ("agora_pottery_chronology_classifier_vit", "Classification", ["vit"]),
        ("agora_pottery_chronology_classifier_tfidf_vit", "Classification", ["tfidf", "vit"]),
        # regression
        ("agora_pottery_chronology_regressor_tfidf", "Regression", ["tfidf"]),
        ("agora_pottery_chronology_regressor_vit", "Regression", ["vit"]),
        ("agora_pottery_chronology_regressor_tfidf_vit", "Regression", ["tfidf", "vit"]),
    ]

    TARGETS = {
        "Classification": ["Historical Period"],
        "Regression": ["Start Year", "Year Range"],
    }

    training_run = get_current_training_run(db)
    train_sample_size = get_train_sample_size(db, training_run.id)

    counter = {
        "models": 0,
        "models_use_feature_sets": 0,
        "models_have_targets": 0,
        "model_versions": 0
    }

    for name, task_name, feature_types in MODELS:
        task = db.query(Task).filter_by(name=task_name).one()

        hf_repo_id = f"dimitriskr/{name}"

        # ─────────────────────────────
        # Model
        # ─────────────────────────────
        model = db.query(Model).filter_by(name=name).first()

        if not model:
            model = Model(
                name=name,
                task_id=task.id,
                hf_repo_id=hf_repo_id,
            )
            db.add(model)
            db.flush()  # ensures model.id exists

            counter["models"] += 1

        # ─────────────────────────────
        # Model ↔ FeatureSets
        # ─────────────────────────────
        for feature_type in feature_types:
            fs = db.query(FeatureSet).filter_by(feature_type=feature_type).one()

            exists = (
                db.query(ModelUsesFeatureSet)
                .filter_by(model_id=model.id, feature_set_id=fs.id)
                .first()
            )

            if not exists:
                db.add(
                    ModelUsesFeatureSet(
                        model_id=model.id,
                        feature_set_id=fs.id,
                    )
                )

                counter["models_use_feature_sets"] += 1

        # ─────────────────────────────
        # Model ↔ Targets
        # ─────────────────────────────
        for target_name in TARGETS[task_name]:
            target = db.query(Target).filter_by(name=target_name).one()

            exists = (
                db.query(ModelHasTarget)
                .filter_by(model_id=model.id, target_id=target.id)
                .first()
            )

            if not exists:
                db.add(
                    ModelHasTarget(
                        model_id=model.id,
                        target_id=target.id,
                    )
                )

                counter["models_have_targets"] += 1

        # ─────────────────────────────
        # Model Version (v1)
        # ─────────────────────────────
        version = "v1"

        mv = db.query(ModelVersion).filter_by(
            model_id=model.id,
            version=version
        ).one_or_none()

        if not mv:
            metadata = load_metadata_from_hf(model.hf_repo_id, version)

            mv = ModelVersion(
                model_id=model.id,
                training_run_id=training_run.id,
                version=version,
                is_current=True,
                train_sample_size=train_sample_size,
                train_time=metadata.get("time"),
                val_loss=metadata.get("val_loss"),
                val_accuracy=(
                    metadata.get("scores", {}).get("accuracy", [None])[0]
                    if task_name == "Classification"
                    else None
                ),
                val_mae=(
                    metadata.get("scores", {}).get("mae", [None])[0]
                    if task_name == "Regression"
                    else None
                ),
            )
            db.add(mv)

            counter["model_versions"] += 1

    for table, c in counter.items():
        print_status(table, c)