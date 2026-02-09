from fastapi import HTTPException

from models import DataSource, HistoricalPeriod, PotteryItem

def validate_item_exists(item: PotteryItem | None):
    if item is None:
        raise HTTPException(status_code=404, detail="Pottery Item not found")

def validate_years(start_year: float, end_year: float):
    if start_year is None or end_year is None:
        raise HTTPException(status_code=409, detail="Start year and end year are required")

    if start_year >= end_year:
        raise HTTPException(status_code=409, detail="Start year must be smaller than end year")

def validate_unique_object_id(db, object_id: str, extra_msg:str = ""):
    if not object_id:
        return

    exists = (
        db.query(PotteryItem)
        .filter(PotteryItem.object_id == object_id)
        .first()
    )

    if exists:
        raise HTTPException(
            status_code=409,
            detail="A pottery item with this object_id already exists. " + extra_msg
        )

def get_data_source(db, description):
    source = (
        db.query(DataSource)
        .filter(DataSource.description == description)
        .one_or_none()
    )

    if not source:
        source = DataSource(description=description)
        db.add(source)
        db.flush()

    return source

def assign_historical_period(
    db,
    start_year: float,
    end_year: float,
) -> HistoricalPeriod | None:

    periods = db.query(HistoricalPeriod).all()
    if not periods:
        return None

    # Special case: single year
    if start_year == end_year:
        for p in periods:
            if p.limit_lower <= start_year <= p.limit_upper:
                return p
        return None

    max_overlap = 0.0
    best_period = None

    for p in periods:
        overlap_start = max(start_year, p.limit_lower)
        overlap_end = min(end_year, p.limit_upper)

        if overlap_start < overlap_end:
            overlap = overlap_end - overlap_start
            if overlap > max_overlap:
                max_overlap = overlap
                best_period = p

    return best_period
