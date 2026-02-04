from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File, Query
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import joinedload

from database import db_dependency
from models import ChronologyPrediction, HistoricalPeriod, ModelVersion, PotteryItem
from schemas import PredictionResponse, ChronologyPredictionSchema, PaginatedResponse
from services import auth_dependency, validate_input, get_feature_types, select_model, load_model, load_target_decoder, \
    extract_features, predict_single, upload_prediction_image, create_prediction_record, save_tmp_file, \
    delete_prediction_image, match_expression

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("", response_model=ChronologyPredictionSchema)
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
    db_model, db_model_version = select_model(db, task, feature_types)

    # Save image temporarily
    image_tmp_path = save_tmp_file(image) if image else None

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
    if image_tmp_path:
        image_tmp_path.unlink()

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    # return PredictionResponse(
    #     prediction=prediction,
    #     breakdown=breakdown,
    #     model=db_model.name,
    #     model_version=db_model_version.version,
    #     feature_types=feature_types,
    # )

    return prediction_record


@router.get("", response_model=PaginatedResponse[ChronologyPredictionSchema])
async def get_predictions(
        db: db_dependency,
        user: auth_dependency,

        input_type: Optional[Literal["text", "image", "text_image"]] = Query(None),
        output_type: Optional[Literal["historical_period", "years"]] = Query(None),
        status: Optional[Literal["pending", "validated"]] = Query(None),
        match: Optional[Literal["exact", "close", "none", "unknown"]] = Query(None),

        sort_by: Literal["created_at", "id"] = Query("created_at"),
        order: Literal["asc", "desc"] = Query("desc"),

        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
):
    query = (db.query(ChronologyPrediction)
    # SQL joins
    .join(ChronologyPrediction.pottery_item, isouter=True)
    .join(PotteryItem.chronology_label, isouter=True)

    # Eager loading
    .options(
        joinedload(ChronologyPrediction.model_version).joinedload(ModelVersion.model),
        joinedload(ChronologyPrediction.historical_period),
        joinedload(ChronologyPrediction.pottery_item).joinedload(PotteryItem.chronology_label),
    ))

    if input_type == "text":
        query = query.filter(
            ChronologyPrediction.input_text.isnot(None),
            ChronologyPrediction.input_image_path.is_(None),
        )

    elif input_type == "image":
        query = query.filter(
            ChronologyPrediction.input_text.is_(None),
            ChronologyPrediction.input_image_path.isnot(None),
        )

    elif input_type == "text_image":
        query = query.filter(
            ChronologyPrediction.input_text.isnot(None),
            ChronologyPrediction.input_image_path.isnot(None),
        )

    if output_type == "historical_period":
        query = query.filter(
            ChronologyPrediction.historical_period_id.isnot(None)
        )

    elif output_type == "years":
        query = query.filter(
            ChronologyPrediction.start_year.isnot(None)
        )

    if status:
        query = query.filter(ChronologyPrediction.status == status)

    if match:
        match_expr = match_expression()
        query = query.filter(match_expr == match)

    total = query.count()

    sort_col = ChronologyPrediction.created_at if sort_by == "created_at" else ChronologyPrediction.id
    if order == "asc":
        query = query.order_by(sort_col.asc(), ChronologyPrediction.id.asc())
    else:
        query = query.order_by(sort_col.desc(), ChronologyPrediction.id.desc())

    items = (
        query
        .limit(limit)
        .offset(offset)
        .all()
    )

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{prediction_id}", response_model=ChronologyPredictionSchema)
async def get_prediction(
        prediction_id: int,
        db: db_dependency,
        user: auth_dependency,
):
    try:
        prediction = (db.query(ChronologyPrediction)
                      .options(
            joinedload(ChronologyPrediction.model_version).joinedload(ModelVersion.model),
            joinedload(ChronologyPrediction.historical_period),
            joinedload(ChronologyPrediction.pottery_item).joinedload(PotteryItem.chronology_label),
        )
                      .filter_by(id=prediction_id)
                      .one())
        return prediction
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Prediction not found")


@router.delete("/{prediction_id}", status_code=204)
def delete_prediction(
        prediction_id: int,
        db: db_dependency,
        user: auth_dependency,
):
    prediction = (
        db.query(ChronologyPrediction)
        .filter(ChronologyPrediction.id == prediction_id)
        .first()
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    # Optional: delete image from HF (see below)
    delete_prediction_image(prediction.input_image_path)

    db.delete(prediction)
    db.commit()

    return None
