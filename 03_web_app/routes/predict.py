from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File
from sqlalchemy.orm import Session
from sympy.stats.rv import probability
from torch.onnx.symbolic_opset9 import tensor

from ML import predict_periods_single, predict_years_single
from database import db_dependency
from schemas import PredictionResponse
from services import auth_dependency, validate_input, get_feature_types, select_model, load_model, load_target_decoder, \
    extract_features

router = APIRouter(prefix="", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
        db: db_dependency,
        user: auth_dependency,

        task: Literal["classification", "regression"] = Form(...),
        text: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
):
    # Validation
    validate_input(text, image)

    feature_types = get_feature_types(text, image)

    db_model, db_model_version = select_model(db, task, feature_types)

    features = extract_features(db, feature_types, text, image)
    feature_list = [feature_tensor for feature_tensor in features.values()]

    model = load_model(db_model.hf_repo_id, db_model_version.version)

    decoder = load_target_decoder(db_model.hf_repo_id, db_model_version.version, task)

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
                "std": int(round(y_std[i])),
                "ci_lower": int(round(y_pred[i] - z * y_std[i])),
                "ci_upper": int(round(y_pred[i] + z * y_std[i])),
            } for i, target in enumerate(targets)
        }

    return PredictionResponse(
        prediction=prediction,
        breakdown=breakdown,
        model=db_model.name,
        model_version=db_model_version.version,
        feature_types=feature_types,
    )
