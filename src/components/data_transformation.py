from src.exception import CustomException
from src.logger import logging
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.externals import joblib

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]  # Update according to your dataset
            categorical_columns = [
                "gender", 
                "race_ethnicity", 
                "parental_level_of_education", 
                "lunch", 
                "test_preparation_course"
            ]

            logging.info("Defining numerical and categorical pipelines")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Combining pipelines into a ColumnTransformer")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating data transformer object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            # Get preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_columns_name = "math_score"  # Update with your target variable
            numerical_columns = ["writing_score", "reading_score"]

            # Split input and target features for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_columns_name], axis=1)
            target_feature_train_df = train_df[target_columns_name]

            input_feature_test_df = test_df.drop(columns=[target_columns_name], axis=1)
            target_feature_test_df = test_df[target_columns_name]

            logging.info("Applying preprocessing object on training and test dataframes")

            # Transform input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input and target features into final arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            # Save the preprocessing object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)

            logging.info("Preprocessing object saved successfully")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)
