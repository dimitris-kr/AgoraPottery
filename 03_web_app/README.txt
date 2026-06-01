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

Copy files to HF Space Repo from original
deploy_sync.py: script that copies only necessary files + reports files to be created/overwritten/deleted in the clone repo based on changes in the source and asks for confirmation before applying changes to clone.