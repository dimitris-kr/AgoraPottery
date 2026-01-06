from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File

from database import db_dependency
from schemas import PredictionResponse
from services import auth_dependency, validate_input, get_feature_types, select_model, load_model, load_target_decoder, \
    extract_features, predict_single

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
    validate_input(task, text, image)

    feature_types = get_feature_types(text, image)

    db_model, db_model_version = select_model(db, task, feature_types)

    features = extract_features(db, feature_types, text, image)
    feature_list = [feature_tensor for feature_tensor in features.values()]

    model = load_model(db_model.hf_repo_id, db_model_version.version)

    decoder = load_target_decoder(db_model.hf_repo_id, db_model_version.version, task)

    prediction, breakdown = predict_single(task, model, feature_list, decoder)

    return PredictionResponse(
        prediction=prediction,
        breakdown=breakdown,
        model=db_model.name,
        model_version=db_model_version.version,
        feature_types=feature_types,
    )
