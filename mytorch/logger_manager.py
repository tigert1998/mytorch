import logging
import sys


def _config_logger(filename):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if filename is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(filename)s:%(lineno)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    root.handlers = [handler]
    return root


class LoggerManager:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = LoggerManager()
        return cls._instance

    def __init__(self):
        self.logger: logging.Logger = _config_logger(None)
