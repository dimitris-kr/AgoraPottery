import io
import shutil
from pathlib import Path

from fastapi import UploadFile
from huggingface_hub import hf_hub_download, upload_file, HfApi

from services import generate_image_path, save_tmp_file


def download_tfidf_vectorizer(repo_id, version):
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{version}/tfidf_vectorizer.joblib"
    )

def download_model(repo_id, version):
    return hf_hub_download(
        repo_id=repo_id,
        filename=f"{version}/model.pt",
        repo_type="model"
    )

def download_model_config(repo_id, version):
    return hf_hub_download(
        repo_id=repo_id,
        filename=f"{version}/config.json",
        repo_type="model"
    )

def download_y_encoder(repo_id, version):
    return hf_hub_download(
        repo_id=repo_id,
        filename=f"{version}/y_encoder.pkl",
        repo_type="model"
    )

def download_y_scaler(repo_id, version):
    return hf_hub_download(
        repo_id=repo_id,
        filename=f"{version}/y_scaler.pkl",
        repo_type="model"
    )


HF_IMAGES_REPO = "dimitriskr/agora_pottery_images"

def upload_prediction_image(image_tmp_path: Path) -> str:
    path_in_repo = generate_image_path(image_tmp_path.suffix, root="predictions")

    upload_file(
        path_or_fileobj=image_tmp_path,
        repo_id=HF_IMAGES_REPO,
        path_in_repo=path_in_repo,
        repo_type="dataset",
    )

    return path_in_repo


def delete_prediction_image(path_in_repo: str | None):
    if not path_in_repo:
        return

    api = HfApi()
    api.delete_file(
        repo_id=HF_IMAGES_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"Delete prediction image {path_in_repo}",
    )

TMP_DIR = Path("./tmp")
TMP_DIR.mkdir(exist_ok=True)

def download_image_tmp(hf_path: str) -> Path:
    local_path = hf_hub_download(
        repo_id=HF_IMAGES_REPO,
        filename=hf_path,
        repo_type="dataset"
    )

    tmp_path = TMP_DIR / Path(hf_path).name
    shutil.copy(local_path, tmp_path)

    return tmp_path