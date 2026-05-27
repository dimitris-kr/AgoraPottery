"""
modal_app.py
------------
The ONLY Modal-aware file in the project. It defines the Modal App and
image, then decorates `run_training` (the plain orchestration function in
retrain.py) for cloud execution.

Deploy:
    modal deploy Modal/modal_app.py

Production call site (services/retrain_service.py):
    modal.Function.lookup("agora-pottery-retrain", "run_training").spawn(payload)
"""

import modal

from Modal.retrain import run_training as _run_training_body


app = modal.App("agora-pottery-retrain")

# Library versions pinned to match the local conda env (environment.yml).
#
# add_local_python_source mounts both packages into the container at their
# natural Python locations, so `from Modal.retrain import ...` and
# `from ML.PotteryChronologyPredictor import ...` both work inside the
# container as they do locally. (The /app/ fallback in
# retrain._import_predictor is unused with this setup but kept as a safety net.)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.45.2",
        "huggingface_hub==0.24.6",
        "scikit-learn==1.6.1",
        "joblib==1.4.2",
        "numpy==2.2.4",
        "requests==2.32.3",
        "Pillow==11.1.0",
    )
    .add_local_python_source("Modal", "ML")
)

hf_secret      = modal.Secret.from_name("huggingface-secret")
webhook_secret = modal.Secret.from_name("webhook-secret")


# Decorate the plain orchestration body. No wrapper function — the decorator
# returns a Modal Function pointing at the same body, registered under the
# name "run_training" (which is how retrain_service looks it up).
run_training = app.function(
    image=image,
    secrets=[hf_secret, webhook_secret],
    gpu="T4",
    timeout=3600,
    retries=0,
    name="run_training",
)(_run_training_body)
