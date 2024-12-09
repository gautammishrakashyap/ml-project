from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import os

def main():
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_obj_file_path)

    except Exception as e:
        logging.error(f"Error in execution: {e}")
        raise CustomException(e)

if __name__ == "__main__":
    main()
