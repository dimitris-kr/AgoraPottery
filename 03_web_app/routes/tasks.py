from fastapi import APIRouter, Query

from database import db_dependency
from models import Task
from schemas import PaginatedResponse, TaskSchema
from services import auth_dependency

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get("", response_model=PaginatedResponse[TaskSchema])
def get_tasks(
        db: db_dependency,
        user: auth_dependency,

        # Pagination
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
):
    query = db.query(Task)

    total = query.count()

    items = (
        query
        .order_by(Task.id)
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

@router.get("/all", response_model=list[TaskSchema])
def get_all_tasks(
        db: db_dependency,
        user: auth_dependency,
):
    query = db.query(Task)

    items = (
        query
        .order_by(Task.id)
        .all()
    )

    return items