import os
import glob
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_MODEL
)

# --- Initialize SentenceTransformer ---
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
dim = embedding_model.get_sentence_embedding_dimension()

# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX)

# --- Load documents from /data folder ---
def load_documents():
    docs = []

    # TXT and MD
    for file in glob.glob("data/*.txt") + glob.glob("data/*.md"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append(f.read())

    # PDFs
    for file in glob.glob("data/*.pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        docs.append(text)

    return docs

# --- Process documents ---
documents = load_documents()
vectors = []
for i, doc in enumerate(documents):
    emb = embedding_model.encode(doc, convert_to_numpy=True).tolist()

    vectors.append({
        "id": f"doc-{i}",
        "values": emb,
        "metadata": {"text": doc[:200]}  # just preview
    })

# Upsert into Pinecone
index.upsert(vectors=vectors)

print(f"✅ {len(documents)} documents added to Pinecone index '{PINECONE_INDEX}' using {EMBEDDING_MODEL}")
