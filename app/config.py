from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# os.environ["HF_HOME"] = os.getenv("HF_HOME")
# os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE")
# os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME")

#project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

#database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

#backend configuration
BACKEND_URL = "http://127.0.0.1:8000"