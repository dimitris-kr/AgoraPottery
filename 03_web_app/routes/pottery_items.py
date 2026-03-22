from typing import Optional, Literal

from fastapi import APIRouter, Query, HTTPException
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import joinedload

from database import db_dependency
from models import PotteryItem, ChronologyLabel, TrainingRun
from schemas import PotteryItemSearchSchema, PotteryItemSchema, PaginatedResponse
from services import auth_dependency
from services.data_service import validate_item_exists

router = APIRouter(prefix="/pottery-items", tags=["Pottery Items"])


@router.get("/search", response_model=list[PotteryItemSearchSchema])
def search_pottery_items(
        db: db_dependency,
        user: auth_dependency,

        q: str = Query(..., min_length=2),
):
    return (
        db.query(PotteryItem)
        .filter(PotteryItem.object_id.ilike(f"%{q}%"))
        .order_by(PotteryItem.id.asc())
        .limit(10)
        .all()
    )


@router.get("/{pottery_item_id}", response_model=PotteryItemSchema)
async def get_pottery_item(
        pottery_item_id: int,
        db: db_dependency,
        user: auth_dependency,
):
    pottery_item = (db.query(PotteryItem)
                    .options(
        joinedload(PotteryItem.chronology_label),
        joinedload(PotteryItem.data_source))
                    .filter_by(id=pottery_item_id)
                    .one_or_none())

    validate_item_exists(pottery_item)

    return pottery_item

@router.get("", response_model=PaginatedResponse[PotteryItemSchema])
async def get_pottery_items(
    db: db_dependency,
    user: auth_dependency,

    # Filters
    historical_period_id: Optional[int] = Query(None),
    start_year: Optional[float] = Query(None),
    end_year: Optional[float] = Query(None),
    data_source_id: Optional[int] = Query(None),
    in_train_set: Optional[bool] = Query(None),

    # Sorting
    sort_by: Literal[
        "created_at",
        "id",
        "object_id",
        "description",
        "start_year",
        "end_year"
    ] = Query("created_at"),

    order: Literal["asc", "desc"] = Query("desc"),

    # Pagination
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    query = (
        db.query(PotteryItem)
        # SQL joins (for filtering/sorting)
        .join(PotteryItem.chronology_label, isouter=True)
        .join(PotteryItem.data_source, isouter=True)
        .join(PotteryItem.in_training_run, isouter=True)

        # Eager loading (for response)
        .options(
            joinedload(PotteryItem.chronology_label)
                .joinedload(ChronologyLabel.historical_period),
            joinedload(PotteryItem.data_source),
        )
    )

    if historical_period_id:
        query = query.filter(
            ChronologyLabel.historical_period_id == historical_period_id
        )

    if start_year is not None:
        query = query.filter(ChronologyLabel.start_year >= start_year)

    if end_year is not None:
        query = query.filter(ChronologyLabel.end_year <= end_year)

    if data_source_id:
        query = query.filter(PotteryItem.data_source_id == data_source_id)

    if in_train_set is not None:
        current_run = (
            db.query(TrainingRun)
            .filter(TrainingRun.is_current == True)
            .one_or_none()
        )

        if current_run:
            if in_train_set:
                pass

    total = query.count()

    sort_map = {
        "created_at": PotteryItem.created_at,
        "id": PotteryItem.id,
        "object_id": PotteryItem.object_id,
        "description": PotteryItem.description,
        "start_year": ChronologyLabel.start_year,
        "end_year": ChronologyLabel.end_year,
    }

    sort_col = sort_map[sort_by]

    if order == "asc":
        query = query.order_by(sort_col.asc(), PotteryItem.id.asc())
    else:
        query = query.order_by(sort_col.desc(), PotteryItem.id.desc())

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
