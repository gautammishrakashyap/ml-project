import sys

class CustomException(Exception):
    def __init__(self, message, sys):
        self.message = message
        self.sys = sys

    def __str__(self):
        return f"Error occurred: {self.message} - {self.sys.exc_info()}"
