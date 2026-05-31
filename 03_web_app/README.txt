03_web_app/
│
├── main.py                      # FastAPI entry point
├── database.py                  # SQLAlchemy session + engine setup
├── environment.yml              # conda environment (dependencies)
├── .env                         # secrets & config
│
├── models/                      # Database models (SQLAlchemy)
│   ├── __init__.py
│   └── user.py
│
├── routes/                      # API endpoints (FastAPI routers)
│   ├── __init__.py
│   ├── auth.py
│
└── services/                    # Non-route logic (functions)
    ├── __init__.py
    ├── auth_service.py


migrations:

alembic init alembic

alembic revision --autogenerate -m "create initial tables"  --rev-id 20251130_1232

alembic upgrade head

after alembic downgrade:
alembic stamp <revision> === “The database structure matches migration f5e246047393 — trust me.”

list environment dependencies:
conda env export --no-builds > environment.yml

Run fastapi:
uvicorn main:app --reload

Deploy to Hugging Face Space with GIT:

- Run once:
  Add new remote to local repo named "hf":
  >> git remote add hf-space <hf-space-repo-url>

- Run every time to deploy to HF Space Repo (after having run >> git commit + >> git push to GitHub)
  >> deploy_web_app.bat
  batch file contains all necessary git commands to deply the web app