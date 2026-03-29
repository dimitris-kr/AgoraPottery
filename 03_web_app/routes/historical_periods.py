from fastapi import APIRouter, Query

from database import db_dependency
from models import HistoricalPeriod
from schemas import PaginatedResponse, HistoricalPeriodSchema
from services import auth_dependency, apply_non_empty_periods_filter

router = APIRouter(prefix="/historical-periods", tags=["Historical Periods"])


@router.get("", response_model=PaginatedResponse[HistoricalPeriodSchema])
def get_periods(
        db: db_dependency,
        user: auth_dependency,

        # Filters
        non_empty: bool = Query(False),

        # Pagination
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
):
    query = db.query(HistoricalPeriod)

    if non_empty:
        query = apply_non_empty_periods_filter(query)

    total = query.count()

    items = (
        query
        .order_by(HistoricalPeriod.limit_lower)
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


@router.get("/all", response_model=list[HistoricalPeriodSchema])
def get_all_periods(
        db: db_dependency,
        user: auth_dependency,

        # Filters
        non_empty: bool = Query(True),
):
    query = db.query(HistoricalPeriod)

    if non_empty:
        query = apply_non_empty_periods_filter(query)

    items = (
        query
        .order_by(HistoricalPeriod.limit_lower)
        .all()
    )

    return items
