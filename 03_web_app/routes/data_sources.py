from fastapi import APIRouter, Query

from database import db_dependency
from models import DataSource
from schemas import PaginatedResponse, DataSourceSchema
from services import auth_dependency, apply_non_empty_sources_filter

router = APIRouter(prefix="/data-sources", tags=["Data Sources"])

@router.get("", response_model=PaginatedResponse[DataSourceSchema])
def get_sources(
        db: db_dependency,
        user: auth_dependency,

        # Filters
        non_empty: bool = Query(False),

        # Pagination
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
):
    query = db.query(DataSource)

    if non_empty:
        query = apply_non_empty_sources_filter(query)

    total = query.count()

    items = (
        query
        .order_by(DataSource.id)
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

@router.get("/all", response_model=list[DataSourceSchema])
def get_all_sources(
        db: db_dependency,
        user: auth_dependency,

        # Filters
        non_empty: bool = Query(True),
):
    query = db.query(DataSource)

    if non_empty:
        query = apply_non_empty_sources_filter(query)

    items = (
        query
        .order_by(DataSource.id)
        .all()
    )

    return items