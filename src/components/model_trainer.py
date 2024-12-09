import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelTrainer:
    def __init__(self):
        pass

    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting data into features and target")

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'SVR': SVR(),
                'XGBoost': xgb.XGBRegressor()
            }

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                mae, mse, rmse = self.train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
                logging.info(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
