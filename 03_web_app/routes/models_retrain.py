from fastapi import APIRouter

from database import db_dependency
from services import auth_dependency
from services.retrain_service import get_eligibility

router = APIRouter(prefix="/models/retrain", tags=["Model Re-Training"])

# ELIGIBILITY

@router.get("/eligibility")
def retrain_eligibility(
    db: db_dependency,
    user: auth_dependency,
):
    """
    Returns whether retraining is available and how many new
    validated items exist outside the current training run.
    """
    return get_eligibility(db)