import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join("artifacts", "dataset")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Entered the Data Transformation method")

        try:
            # Load CSVs
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train Data Shape: {train_df.shape}")
            logging.info(f"Test Data Shape: {test_df.shape}")

            # ======================
            # 1. Map labels to integers
            # ======================
            label2id = {"normal": 0, "offensive": 1, "hatespeech": 2}
            id2label = {v: k for k, v in label2id.items()}

            for df in [train_df, test_df]:
                df["label"] = df["label"].map(label2id)

            # Keep only needed columns
            train_df = train_df[["comment", "label"]]
            test_df = test_df[["comment", "label"]]

            # Split train into train/val
            train_df, val_df = train_test_split(
                train_df,
                test_size=0.1,
                stratify=train_df["label"],
                random_state=42
            )

            # ======================
            # 2. Convert to Hugging Face Datasets
            # ======================
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            test_dataset = Dataset.from_pandas(test_df)

            dataset = DatasetDict({
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            })

            logging.info("✅ Data Transformation completed successfully")

            return dataset, label2id, id2label

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Run Data Ingestion
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    # Run Data Transformation
    transformation = DataTransformation()
    dataset, label2id, id2label = transformation.initiate_data_transformation(train_path, test_path)

    print(dataset)
