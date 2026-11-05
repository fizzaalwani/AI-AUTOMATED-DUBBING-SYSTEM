# import logging
# import os
# from logging.handlers import RotatingFileHandler

# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# LOG_FILE = os.path.join(LOG_DIR, "app.log")

# formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

# file_handler = RotatingFileHandler(
#     LOG_FILE,
#     maxBytes=5 * 1024 * 1024,
#     backupCount=5,
#     encoding="utf-8"
# )
# file_handler.setFormatter(formatter)

# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)

# logger = logging.getLogger("VideoDubbing")
# logger.setLevel(logging.INFO)

# if not logger.handlers:
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")


formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


# UTF-8 file logging
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)


# Force Windows console to use UTF-8
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)


logger = logging.getLogger("VideoDubbing")
logger.setLevel(logging.INFO)


if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)