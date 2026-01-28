import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import requests
from PIL import Image
from io import BytesIO
from services import hf_image_url, auth_dependency

router = APIRouter(prefix="/images", tags=["Images"])

HF_TOKEN = os.getenv("HF_TOKEN")

SIZES = {
    "thumb": 300,
    "medium": 800,
    "full": None
}


@router.get("/{image_path:path}")
def proxy_image(
        image_path: str,
        user: auth_dependency,
        size: str = "full",
):
    url = hf_image_url(image_path)

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    r = requests.get(url, headers=headers, stream=True)

    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(BytesIO(r.content)).convert("RGB")
    w = SIZES.get(size, None)
    if w:
        ratio = w / img.width
        h = int(img.height * ratio)
        img = img.resize((w, h), Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="WEBP", quality=80)
    buffer.seek(0)


    return StreamingResponse(
        buffer,
        media_type="image/webp",
        headers={
            "Cache-Control": "public, max-age=86400"
        }
    )
