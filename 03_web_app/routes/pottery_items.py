from fastapi import APIRouter, Query, HTTPException
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import joinedload

from database import db_dependency
from models import PotteryItem
from schemas import PotteryItemSearchSchema, PotteryItemSchema
from services import auth_dependency

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
    try:
        pottery_item = (db.query(PotteryItem)
                        .options(
            joinedload(PotteryItem.chronology_label),
            joinedload(PotteryItem.data_source))
                        .filter_by(id=pottery_item_id)
                        .one())
        return pottery_item
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Pottery Item not found")


