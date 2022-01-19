import datetime as dt
import logging
import os
import sys
from typing import Optional


def get_logger(logger_path: str,
               name_suffix: Optional[str] = None) -> logging.Logger:
    """Construct logger and store.

    :param str logger_path:
        Path for logger output.
    :param str name_suffix:
        Suffix for logger file.
    :return:
        Logger object.
    :rtype:
        logger.Logger
    """

    if not os.path.exists(logger_path):
        os.makedirs(logger_path, exist_ok=True)

    # Create timestamp and name for logger
    now_string = f"{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    logger_fname = f'{logger_path}/{now_string}'

    logger_fname += f'_{name_suffix}.log' if name_suffix else '.log'

    # Create formatter
    log_format = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
    stream_formatter = logging.Formatter(log_format)

    # Create file formatter
    date_format = '%Y-%m-%d %H:%M:%S'
    file_formatter = logging.Formatter(log_format, datefmt=date_format)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Fix for erroneous logger duplication
    logger.handlers = []

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(filename=logger_fname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(stream_formatter)
    logger.addHandler(console_handler)

    return logger
