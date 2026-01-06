from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ML import predict_periods_single, predict_years_single
from models import Model, Task, ModelUsesFeatureSet, FeatureSet, ModelVersion


def validate_input(task, text, image):
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task}")

    if not text and not image:
        raise HTTPException(
            status_code=400,
            detail="At least one of text or image must be provided",
        )

def get_feature_types(text, image):
    feature_types = []
    if text:
        feature_types.append("tfidf")

    if image:
        feature_types.append("vit")

    return feature_types


def select_model(
    db: Session,
    task_name: str,
    feature_types: list[str],
):
    feature_types = tuple(sorted(feature_types))

    total_fs_sq = (
        db.query(
            ModelUsesFeatureSet.model_id.label("model_id"),
            func.count(FeatureSet.id).label("total_features"),
        )
        .join(FeatureSet)
        .group_by(ModelUsesFeatureSet.model_id)
        .subquery()
    )

    matched_fs_sq = (
        db.query(
            ModelUsesFeatureSet.model_id.label("model_id"),
            func.count(FeatureSet.id).label("matched_features"),
        )
        .join(FeatureSet)
        .filter(FeatureSet.feature_type.in_(feature_types))
        .group_by(ModelUsesFeatureSet.model_id)
        .subquery()
    )

    # Final query: exact match
    models_query = (
        db.query(Model)
        .join(Task)
        .join(total_fs_sq, total_fs_sq.c.model_id == Model.id)
        .join(matched_fs_sq, matched_fs_sq.c.model_id == Model.id)
        .filter(Task.name == task_name.capitalize())
        .filter(total_fs_sq.c.total_features == len(feature_types))
        .filter(matched_fs_sq.c.matched_features == len(feature_types))
    )

    model = models_query.one_or_none()

    if not model:
        raise ValueError(
            f"No model found for task='{task_name}' "
            f"and features={feature_types}"
        )

    # Get current model version
    model_version = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.model_id == model.id,
            ModelVersion.is_current == True,
        )
        .one_or_none()
    )

    if not model_version:
        raise ValueError(
            f"No current model version found for model '{model.name}'"
        )

    return model, model_version


def predict_single(task, model, feature_list, decoder):


    if task == "classification":
        y_pred, y_probs = predict_periods_single(model, feature_list, decoder)
        prediction = y_pred
        breakdown = {
            "probabilities": y_probs
        }
    else:
        y_pred, y_std = predict_years_single(model, feature_list, decoder)

        start_year = int(round(y_pred[0], 0))
        year_range = int(round(y_pred[1], 0))

        end_year = start_year + year_range

        prediction = [start_year, end_year]
        z = 1.96
        targets = ["start_year", "year_range"]
        breakdown = {
            target: {
                "prediction": int(round(y_pred[i], 0)),
                "std": int(round(y_std[i], 0)),
                "ci_lower": int(round(y_pred[i] - z * y_std[i], 0)),
                "ci_upper": int(round(y_pred[i] + z * y_std[i], 0)),
            } for i, target in enumerate(targets)
        }

    return prediction, breakdown