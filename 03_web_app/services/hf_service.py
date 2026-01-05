from huggingface_hub import hf_hub_download

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