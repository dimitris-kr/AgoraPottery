import os
from datetime import datetime, timedelta

PATH_DATA = os.path.abspath(os.path.join(os.getcwd(), "../../data/agora12_data_pp.csv"))

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_STATE = 42
STRATIFY_COLUMN = "HistoricalPeriod"

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6

# ──────────────────────────────────────────────
# SEED TIMESTAMPS
# Reproduce the real development timeline for seeding production DB (on Neon).
# These are the actual dates the work was done (mirrors the original local DB).
# Explicitly provided created_at/updated_at override the func.now() default.
# ──────────────────────────────────────────────

DATE_USERS = datetime(2025, 10, 19, 12, 0, 0)
DATE_POTTERY_ITEMS = datetime(2025, 12, 28, 12, 0, 0)
WINDOW_POTTERY_ITEMS = timedelta(hours=10)
DATE_CHRONOLOGY_LABELS = datetime(2025, 12, 29, 12, 0, 0)
DATE_TRAINING_RUNS_V1 = datetime(2025, 12, 30, 12, 0, 0)
DATE_MODEL_VERSIONS_V1 = datetime(2026, 1, 2, 12, 0, 0)
DATE_CHRONOLOGY_PREDICTIONS = datetime(2026, 1, 11, 12, 0, 0)  # demo test-set predictions
WINDOW_CHRONOLOGY_PREDICTIONS = timedelta(hours=2)
