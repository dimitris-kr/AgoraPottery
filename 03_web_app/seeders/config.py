import os

PATH_DATA = os.path.abspath(os.path.join(os.getcwd(), "../../data/agora12_data_pp.csv"))

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_STATE = 42
STRATIFY_COLUMN = "HistoricalPeriod"

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6