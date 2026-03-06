# Data Paths
DATA_PATH = "data/train.csv"
STORE_STATE = "data/store_states.csv"

# Columns
TARGET_COL = "Sales"
DATE_COL = "Date"
CATEGORICAL_COLS = ["Store", "DayOfWeek", "day", "month", "year", "Promo", "State"]

# Data Preparing Configuration
SPLIT_TYPE = "random"
TEST_SIZE = 0.1

# Model Configuration
BATCH_SIZE = 128
LOSS_TYPE = "mse"
LEARNING_RATE = 1e-3
EPOCHS = 10
OUTPUT_SIGMOID = True
ENSEMBLE_SIZE = 5
EMB_DIMS = {
    "Store": 10,
    "DayOfWeek": 6,
    "day": 10,
    "month": 6,
    "year": 2,
    "Promo": 1,
    "State": 6,
}
RANDOM_STATE = 42