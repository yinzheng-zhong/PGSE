import logging
import colorlog
import os


class Logger:
    def __init__(self, name):
        self.logger = colorlog.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Console handler with colored formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(console_handler)

        # Ensure the log directory exists
        log_dir = './log/'
        os.makedirs(log_dir, exist_ok=True)

        # File handler that appends to the log file
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
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
