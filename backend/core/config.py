# config.py
# .env 파일에서 설정값을 읽어온다
# pip install python-dotenv

from dotenv import load_dotenv
import os
from pathlib import Path


load_dotenv()  # .env 파일을 읽어서 환경변수로 등록

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDF_PATH       = os.getenv("PDF_PATH", "document.pdf")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K          = int(os.getenv("TOP_K", 3))
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL     = os.getenv("LLM_MODEL")
BASE_URL      = os.getenv("BASE_URL")

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

INDEX_PATH = Path(
    os.getenv("INDEX_PATH", DATA_DIR / "embeddings" / "faiss_index")
)

META_PATH = Path(
    os.getenv("META_PATH", INDEX_PATH / "meta.json")
)

DOCS_PATH = Path(
    os.getenv("DOCS_PATH", DATA_DIR / "processed" / "text")
)