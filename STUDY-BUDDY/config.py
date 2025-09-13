# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Any Groq model that supports chat completions (examples: "llama-3.1-70b-versatile", "llama-3.1-8b-instant")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

# --- Pinecone (v5) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "study-buddy-index")
# Serverless spec: choose cloud + region. We accept PINECONE_CLOUD and PINECONE_REGION for clarity.
# If the user only has PINECONE_ENV, we treat it as region.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION") or os.getenv("PINECONE_ENV", "us-east-1")

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Simple validation (defer strict checks until first use to not break /docs)
def require(key: str, value: str):
    if not value:
        raise ValueError(f"{key} not set in environment (.env)")

def validate_critical():
    require("GROQ_API_KEY", GROQ_API_KEY)
    require("PINECONE_API_KEY", PINECONE_API_KEY)
