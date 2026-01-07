from uuid import uuid4
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile
import shutil

def generate_image_path(suffix: str, root: str = "", year=True, month=True) -> str:
    now = datetime.now()
    path = f"{root}/" if root else ""
    if year: path += f"{now.year}/"
    if year and month: path += f"{now.month:02d}/"
    path += f"{uuid4().hex}{suffix}"
    return path

TMP_DIR = Path("./tmp")
TMP_DIR.mkdir(exist_ok=True)

def save_tmp_file(file: UploadFile) -> Path:
    suffix = Path(file.filename).suffix
    tmp_path = TMP_DIR / f"{uuid4().hex}{suffix}"

    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return tmp_path