"""
utils/logger.py
───────────────
Structured logging for the LLM-FMEA framework.
"""
import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_DIR / "llm_fmea.log")
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    logger.addHandler(fh)

    return logger
