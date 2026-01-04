from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File
from sqlalchemy.orm import Session

from database import db_dependency
from schemas import PredictionResponse
from services import auth_dependency, validate_input, get_feature_types, select_model
from services.feature_service import extract_features

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

    model, model_version = select_model(db, task, feature_types)

    features = extract_features(db, feature_types, text, image)
    features = {t: tensor.tolist() for t, tensor in features.items()}
    return PredictionResponse(
        model_name=model.name,
        model_version=model_version.version,
        feature_types=feature_types,
        features=features,
    )

