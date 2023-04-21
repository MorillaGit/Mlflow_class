import os
from pathlib import Path

DATASET_ROOT_PATH = str(
    Path(__file__).parent.parent / "datasets/sprint_2_structured_data"
)
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_TRAIN = str(Path(DATASET_ROOT_PATH) / "application_train_aai.csv")
DATASET_TRAIN_URL = (
    "https://drive.google.com/uc?id=1vT0u2xndjNFIE-gCoW7s0tJuV4aPyz0B&confirm=t"
)
DATASET_TEST = str(Path(DATASET_ROOT_PATH) / "application_test_aai.csv")
DATASET_TEST_URL = (
    "https://drive.google.com/uc?id=1sXWs2hyG34xFzRepf9fOKkCXq4kGjMT5&confirm=t"
)
DATASET_DESCRIPTION = str(
    Path(DATASET_ROOT_PATH) / "HomeCredit_columns_description.csv"
)
DATASET_DESCRIPTION_URL = (
    "https://drive.google.com/uc?id=1e8jbK4zNv95Yd2acycf27MbrDcCwAgFS&confirm=t"
)

DATASET_FILENAME_IMAGE = (
    "C:/Users/crist/Documents/anyoneai/mlflow_class/datasets/eu-car-dataset_subset"
)
