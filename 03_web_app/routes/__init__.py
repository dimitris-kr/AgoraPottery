from fastapi import APIRouter
from .auth import router as auth_router
from .predictions import router as predict_router
from .images import router as images_router
from .pottery_items import router as pottery_items_router
from .historical_periods import router as historical_periods_router
from .data_sources import router as data_sources_router
from .models import router as model_router
from .tasks import router as task_router
router = APIRouter()

# Register all routes
router.include_router(auth_router)
router.include_router(predict_router)
router.include_router(images_router)
router.include_router(pottery_items_router)
router.include_router(historical_periods_router)
router.include_router(data_sources_router)
router.include_router(model_router)
router.include_router(task_router)