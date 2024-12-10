import joblib
import os

def load_object(file_path):
    """
    Load a Python object from the specified file path using joblib.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {str(e)}")
