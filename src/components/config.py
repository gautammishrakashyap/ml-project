import os

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
