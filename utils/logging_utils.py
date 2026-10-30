import logging
import sys

def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level   =getattr(logging, level.upper(), logging.INFO),
        format  =fmt,
        datefmt =datefmt,
        handlers=handlers,
    )