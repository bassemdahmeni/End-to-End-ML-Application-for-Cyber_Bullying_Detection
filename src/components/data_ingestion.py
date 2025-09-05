import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    val_data_path: str = os.path.join('artifacts', "val.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # ⚡ Use forward slashes for Windows compatibility
            df = pd.read_csv(r"C:\Users\el papi\Desktop\CyberBullying\notebooks\data\final_hateXplain.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train/Validation/Test split initiated")
            
            # 1st split: train + temp (val+test)
            train_set, temp_set = train_test_split(df, test_size=0.3, random_state=42)
            # 2nd split: validation + test
            val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, val_data, test_data = obj.initiate_data_ingestion()
    print("✅ Data Ingestion completed!")
    print(f"Train data saved at: {train_data}")
    print(f"Validation data saved at: {val_data}")
    print(f"Test data saved at: {test_data}")
