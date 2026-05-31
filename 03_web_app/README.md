---
title: Pottery Chronology Predictor API
emoji: 🏺
colorFrom: red
colorTo: yellow
sdk: docker
python_version: 3.12
app_port: 7860
pinned: false
short_description: Predict historical period or exact year range of pottery
---

# Pottery Chronology Predictor API

FastAPI backend for the Agora Pottery Chronology Prediction project.

Main Features:
- Chronology prediction for pottery items, with input either their description, image or both (multimodal), and output the estimated historical period (classification) or year range (regression).
- Chronology feedback from users (experts).
- Retraining of predictive models with new items with verified chronology on Modal (Serverless GPU Platform) .

Deployment Specifications:
- This Space runs the API in a Docker container.
- The container listens on port **7860** (HF Spaces default).
- Configuration (secrets, tokens, etc.) is supplied via Space variables & secrets.
