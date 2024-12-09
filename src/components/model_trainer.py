import os
import sys
import joblib
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "SVR": SVR(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "ExtraTreesRegressor": ExtraTreesRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
            }

            best_model = None
            best_r2 = -1
            best_model_name = ""

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = model_name

                logging.info(f"{model_name} R² Score: {r2}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

            logging.info(f"Best Model: {best_model_name} with R² Score: {best_r2}")
            return best_r2, best_model_name, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
