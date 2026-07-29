import logging
from pathlib import Path

class Logger:
    def __init__(self,log_dir="logs",log_file="medical_ie.log"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("MedicalIE")
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            Path(log_dir) / log_file,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_loggers(self):
        return self.logger

if __name__ == "__main__":
    logger = Logger().get_loggers()

    logger.info("Logger initialized successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")