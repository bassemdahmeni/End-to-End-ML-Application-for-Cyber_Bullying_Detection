import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Creates a detailed error message with file name, line number, and the actual error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line: [{line_number}] "
        f"with message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Custom exception class that logs and formats error messages.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message



