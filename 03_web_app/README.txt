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

- Run every time to deploy to HF Space Repo (after having run >> git commit + >> git push origin main # → GitHub)
  1. Create a temporary branch in the repo named "hf-deploy" where 03_wb_app is the root
     >> git subtree split --prefix 03_web_app -b hf-deploy
  2. Push temp branch to remote HF Space Repo
     >> git push hf-space hf-deploy:main --force
  3. Delete temp branch
     >> git branch -D hf-deploy