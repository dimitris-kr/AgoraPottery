from huggingface_hub import hf_hub_download

def download_tfidf_vectorizer(repo_id, version):
    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{version}/tfidf_vectorizer.joblib"
    )
