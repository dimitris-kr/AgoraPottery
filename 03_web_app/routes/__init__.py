from fastapi import APIRouter
from .auth import router as auth_router
from .predictions import router as predict_router
from .images import router as images_router
from .pottery_items import router as pottery_items_router

router = APIRouter()

# Register all routes
router.include_router(auth_router)
router.include_router(predict_router)
router.include_router(images_router)
router.include_router(pottery_items_router)