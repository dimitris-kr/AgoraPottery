from fastapi import APIRouter
from .auth import router as auth_router
from .predict import router as predict_router

router = APIRouter()

# Register all routes
router.include_router(auth_router)
router.include_router(predict_router)