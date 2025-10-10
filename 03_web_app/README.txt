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

alembic revision --autogenerate -m "create initial tables"

alembic upgrade head


list environment dependencies:
conda env export --no-builds > environment.yml
