from fastapi import APIRouter
from .auth import router as auth_router

router = APIRouter()

# Register all routes
router.include_router(auth_router)