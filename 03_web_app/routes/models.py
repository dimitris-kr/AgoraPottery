from typing import Optional, Literal

from fastapi import APIRouter, Query
from sqlalchemy.orm import joinedload

from database import db_dependency
from models import ModelVersion, Model
from schemas import PaginatedResponse, ModelVersionSchema, ModelSchema, TargetScoresSchema
from services import auth_dependency, validate_model_exists, validate_model_version_exists, load_model_scores

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=PaginatedResponse[ModelVersionSchema])
async def get_models(
        db: db_dependency,
        user: auth_dependency,

        # Filters
        task_id: Optional[int] = Query(None),

        # Sorting
        sort_by: Literal[
            "model_id",
            "created_at",
            "version",
            "train_sample_size",
            "val_accuracy",
            "val_mae",
        ] = Query("model_id"),

        order: Literal["asc", "desc"] = Query("asc"),

        # Pagination
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
):
    return await get_model_versions(
        db=db,
        user=user,

        is_current=True,
        task_id=task_id,
        model_id=None,

        sort_by=sort_by,
        order=order,

        limit=limit,
        offset=offset,
    )


@router.get("/versions", response_model=PaginatedResponse[ModelVersionSchema])
async def get_model_versions(
        db: db_dependency,
        user: auth_dependency,

        # Filters
        is_current: Optional[bool] = Query(None),
        task_id: Optional[int] = Query(None),
        model_id: Optional[int] = Query(None),

        # Sorting
        sort_by: Literal[
            "model_id",
            "created_at",
            "version",
            "train_sample_size",
            "val_accuracy",
            "val_mae",
        ] = Query("created_at"),

        order: Literal["asc", "desc"] = Query("desc"),

        # Pagination
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
):
    # Main Query
    query = (
        db.query(ModelVersion)
        .join(ModelVersion.model)
        .join(Model.task)
        .options(
            joinedload(ModelVersion.model).joinedload(Model.task),
            joinedload(ModelVersion.model).joinedload(Model.targets),
            joinedload(ModelVersion.model).joinedload(Model.feature_sets),
        )
    )

    # Filters
    if is_current is not None:
        query = query.filter(ModelVersion.is_current == is_current)

    if model_id:
        query = query.filter(ModelVersion.model_id == model_id)

    if task_id:
        query = query.filter(Model.task_id == task_id)

    # Total Count
    total = query.count()

    # Sort
    sort_map = {
        "model_id": ModelVersion.model_id,
        "created_at": ModelVersion.created_at,
        "version": ModelVersion.version,
        "train_sample_size": ModelVersion.train_sample_size,
        "val_accuracy": ModelVersion.val_accuracy,
        "val_mae": ModelVersion.val_mae,
    }

    sort_col = sort_map[sort_by]

    if order == "asc":
        query = query.order_by(sort_col.asc(), ModelVersion.id.asc())
    else:
        query = query.order_by(sort_col.desc(), ModelVersion.id.desc())

    # Results
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


@router.get("/{model_id}", response_model=ModelSchema)
async def get_model(
        db: db_dependency,
        user: auth_dependency,
        model_id: int,
):
    model = (db.query(Model)
             .options(
        joinedload(Model.task),
        joinedload(Model.targets),
        joinedload(Model.feature_sets),
    )
             .filter_by(id=model_id)
             .one_or_none()
             )

    validate_model_exists(model)

    return model


@router.get("/{model_id}/versions", response_model=list[ModelVersionSchema])
async def get_single_model_versions(
        db: db_dependency,
        user: auth_dependency,
        model_id: int,
):
    # Main Query
    query = (
        db.query(ModelVersion)
        .options(
            joinedload(ModelVersion.model).joinedload(Model.task),
            joinedload(ModelVersion.model).joinedload(Model.targets),
            joinedload(ModelVersion.model).joinedload(Model.feature_sets),
        )
        .filter_by(model_id=model_id)
    )

    query = query.order_by(ModelVersion.created_at.asc())

    return query.all()


@router.get("/{model_id}/scores", response_model=list[TargetScoresSchema])
async def get_model_current_version_scores(
        db: db_dependency,
        user: auth_dependency,
        model_id: int,
):
    model_version = (
        db.query(ModelVersion)
        .options(
            joinedload(ModelVersion.model).joinedload(Model.task),
        )
        .filter(
            ModelVersion.model_id == model_id,
            ModelVersion.is_current == True
        )
        .one_or_none()
    )

    validate_model_version_exists(model_version)

    return load_model_scores(
        model_version.model.hf_repo_id,
        model_version.version,
        model_version.model.task.name
    )
