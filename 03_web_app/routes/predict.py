from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File

from database import db_dependency
from models import ChronologyPrediction, HistoricalPeriod
from schemas import PredictionResponse
from services import auth_dependency, validate_input, get_feature_types, select_model, load_model, load_target_decoder, \
    extract_features, predict_single, upload_prediction_image, create_prediction_record, save_tmp_file

router = APIRouter(prefix="", tags=["Prediction"])


@router.post("/predict")
async def predict(
        db: db_dependency,
        user: auth_dependency,

        task: Literal["classification", "regression"] = Form(...),
        text: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
):
    # Validation
    validate_input(task, text, image)

    # Determine Feature Types and Model
    feature_types = get_feature_types(text, image)

    image_tmp_path = save_tmp_file(image) if image else None

    db_model, db_model_version = select_model(db, task, feature_types)

    # Extract Features
    features = extract_features(db, feature_types, text, image_tmp_path)
    feature_list = [feature_tensor for feature_tensor in features.values()]

    # Load Model
    model = load_model(db_model.hf_repo_id, db_model_version.version)

    decoder = load_target_decoder(db_model.hf_repo_id, db_model_version.version, task)

    # Predict
    prediction, breakdown = predict_single(task, model, feature_list, decoder)

    # Save Prediction
    image_path = upload_prediction_image(image_tmp_path) if image else None
    prediction_record = create_prediction_record(db, task, text, image_path, prediction, breakdown, db_model_version)
    image_tmp_path.unlink()

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return PredictionResponse(
        prediction=prediction,
        breakdown=breakdown,
        model=db_model.name,
        model_version=db_model_version.version,
        feature_types=feature_types,
    )
