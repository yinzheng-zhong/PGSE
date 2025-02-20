import logging
import colorlog
import os

class Logger:
    LEVEL_MAP = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }

    def __init__(self, name, verbosity=2):
        """
        :param name: Name of the logger (typically __name__ or any string identifier).
        :param verbosity: Logging level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG).
                          Defaults to 2 (INFO).
        """
        # Determine logging level from the provided verbosity
        log_level = self.LEVEL_MAP.get(verbosity, logging.INFO)

        # Create logger with specified name
        self.logger = colorlog.getLogger(name)
        self.logger.setLevel(log_level)

        # Console handler with colored formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(console_handler)

        # Ensure the log directory exists
        log_dir = './log/'
        os.makedirs(log_dir, exist_ok=True)

        # File handler that appends to the log file
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
