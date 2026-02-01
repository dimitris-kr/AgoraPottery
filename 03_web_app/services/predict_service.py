from typing import Literal

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import case, func, literal, Float

from ML import predict_periods_single, predict_years_single
from models import Model, Task, ModelUsesFeatureSet, FeatureSet, ModelVersion, ChronologyPrediction, HistoricalPeriod, \
    ChronologyLabel


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

def create_prediction_record(db, task, text, image_path, prediction, breakdown, db_model_version, status="pending"):
    prediction_record = ChronologyPrediction(
        model_version_id=db_model_version.id,
        input_text=text,
        input_image_path=image_path,
        breakdown=breakdown,
        status=status,
    )

    if task == "classification":
        period = (
            db.query(HistoricalPeriod)
            .filter(HistoricalPeriod.name == prediction)
            .one()
        )
        prediction_record.historical_period_id = period.id

    else:
        start_year, end_year = prediction
        prediction_record.start_year = start_year
        prediction_record.end_year = end_year
        prediction_record.midpoint_year = (start_year + end_year) / 2
        prediction_record.year_range = end_year - start_year

    return prediction_record


# MATCH: Calculated Field
# Compare true vs predicted chronology

EXACT_OVERLAP = 0.9        # 90% overlap
CLOSE_OVERLAP = 0.4        # meaningful overlap
CLOSE_MIDPOINT = 50        # years

def overlap_ratio(a_start, a_end, b_start, b_end):
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0

def midpoint_distance(a_midpoint, b_midpoint):
    return abs(a_midpoint - b_midpoint)

def match_regression(pred, true) -> Literal["exact", "close", "none"]:
    overlap = overlap_ratio(
        pred.start_year, pred.end_year,
        true.start_year, true.end_year
    )

    midpoint_diff = midpoint_distance(
        pred.midpoint_year,
        true.midpoint_year
    )

    if overlap >= EXACT_OVERLAP:
        return "exact"

    if overlap >= CLOSE_OVERLAP or midpoint_diff <= CLOSE_MIDPOINT:
        return "close"

    return "none"

# def compute_match(prediction: ChronologyPredictionSchema) -> Literal["exact", "close", "none", "unknown"]:
#     if not prediction.pottery_item or not prediction.pottery_item.chronology_label:
#         return "unknown"
#
#     true = prediction.pottery_item.chronology_label
#
#     if prediction.historical_period_id:
#         return (
#             "exact"
#             if prediction.historical_period_id == true.historical_period_id
#             else "none"
#         )
#
#     return match_regression(prediction, true)


def match_expression():
    pred = ChronologyPrediction
    true = ChronologyLabel

    # intersection
    intersection = func.greatest(
        0,
        func.least(pred.end_year, true.end_year)
        - func.greatest(pred.start_year, true.start_year)
    )

    # union
    union = (
        func.greatest(pred.end_year, true.end_year)
        - func.least(pred.start_year, true.start_year)
    )

    overlap_ratio = intersection / func.nullif(union, 0)

    midpoint_diff = func.abs(
        pred.midpoint_year - true.midpoint_year
    )

    return case(
        # ─────────────────────────────
        # Unknown (no true label)
        # ─────────────────────────────
        (
            true.id.is_(None),
            literal("unknown"),
        ),

        # ─────────────────────────────
        # Classification
        # ─────────────────────────────
        (
            pred.historical_period_id.isnot(None),
            case(
                (
                    pred.historical_period_id == true.historical_period_id,
                    literal("exact"),
                ),
                else_=literal("none"),
            ),
        ),

        # ─────────────────────────────
        # Regression – exact
        # ─────────────────────────────
        (
            overlap_ratio >= EXACT_OVERLAP,
            literal("exact"),
        ),

        # ─────────────────────────────
        # Regression – close
        # ─────────────────────────────
        (
            (overlap_ratio >= CLOSE_OVERLAP)
            | (midpoint_diff <= CLOSE_MIDPOINT),
            literal("close"),
        ),

        # ─────────────────────────────
        # Default
        # ─────────────────────────────
        else_=literal("none"),
    )
