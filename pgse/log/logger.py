import logging
import os
from typing import Optional

import colorlog


class Logger:
    LEVEL_MAP = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }

    def __init__(self, name: str, verbosity: int = 2, log_file: Optional[str] = None) -> None:
        """
        :param name: Name of the logger (typically __name__ or any string identifier).
        :param verbosity: Logging level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG).
                          Defaults to 2 (INFO).
        :param log_file: Path of a file to append the log to. Nothing is written to disk
                         when it is left unset.
        """
        # Determine logging level from the provided verbosity
        self.level = self.LEVEL_MAP.get(verbosity, logging.INFO)

        # Create logger with specified name
        self.logger = colorlog.getLogger(name)
        self.logger.setLevel(self.level)
        # Handlers are attached here, so the records must not also reach the root
        # logger of the application that imported PGSE.
        self.logger.propagate = False

        # Console handler with colored formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(console_handler)

        if log_file:
            self.add_file_handler(log_file)

    def add_file_handler(self, log_file: str) -> None:
        """
        Also append the log to a file, creating its directory if needed.

        :param log_file: Path of the file to append to.
        """
        directory = os.path.dirname(log_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(self.level)
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
