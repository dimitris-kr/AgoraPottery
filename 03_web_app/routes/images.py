import os
import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import requests
from PIL import Image
from io import BytesIO
from services import hf_image_url, auth_dependency

router = APIRouter(prefix="/images", tags=["Images"])

HF_TOKEN = os.getenv("HF_TOKEN")

SIZES = {
    "thumb": 300,
    "medium": 800,
    "full": None,
}

# Server-side cache for RESIZED images (thumb/medium) — NOT full.
# Rationale:
#  - thumbnails are requested in bursts (data tables, many at once) and
#    are tiny, so caching them is high-value / low-space
#  - full images are large and only requested one at a time (single-item views),
#    so caching them would grow the cache fast for little benefit
#  - dataset grows with every prediction upload, so keep the cached footprint small
CACHE_DIR = Path("./tmp/img_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_HEADERS = {"Cache-Control": "public, max-age=86400"} # browser cache


def _cache_path(image_path: str, size: str) -> Path:
    key = hashlib.sha256(f"{size}:{image_path}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.webp"


@router.get("/{image_path:path}")
def proxy_image(
        image_path: str,
        user: auth_dependency,
        size: str = "full",
):
    if size not in SIZES:
        raise HTTPException(status_code=400, detail=f"Invalid size: {size}")

    width = SIZES[size]

    # Search & serve straight from disk if size is not 'full' / width is not None
    cache_file = None
    if width is not None:
        cache_file = _cache_path(image_path, size)
        if cache_file.exists():
            return FileResponse(cache_file, media_type="image/webp", headers=CACHE_HEADERS)

    # Download original from HF
    url = hf_image_url(image_path)
    r = requests.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(BytesIO(r.content)).convert("RGB")

    # Resize if size is not 'full' / width is not None
    if width is not None:
        ratio = width / img.width
        h = int(img.height * ratio)
        img = img.resize((width, h), Image.LANCZOS)

    # Resized → save to cache and serve the file
    if cache_file is not None:
        img.save(cache_file, format="WEBP", quality=80)
        return FileResponse(cache_file, media_type="image/webp", headers=CACHE_HEADERS)

    # Full → serve without caching (stream from memory)
    buffer = BytesIO()
    img.save(buffer, format="WEBP", quality=80)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/webp", headers=CACHE_HEADERS)
