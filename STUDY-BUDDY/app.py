from flask import Flask, request, jsonify, render_template
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import uuid
import json

# =========================
# 📌 Load environment variables
# =========================
load_dotenv()

app = Flask(__name__)

# Load API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "study-buddy"

# =========================
# 📌 Init Pinecone
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it does not exist
if PINECONE_INDEX not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created new Pinecone index: {PINECONE_INDEX}")
else:
    print(f"ℹ️ Pinecone index '{PINECONE_INDEX}' already exists.")

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX)

# =========================
# 📌 Models
# =========================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


# =========================
# 📌 CHAT ENDPOINT
# =========================
@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_question = data.get("question")

    try:
        # Embed query
        query_vector = embed_model.encode(user_question).tolist()
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)

        # Collect context
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

        final_prompt = f"""Use the context below to answer the question clearly.

Context:
{context}

Question: {user_question}
Answer:"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": final_prompt}]
        )

        return jsonify({"answer": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 📌 QUIZ ENDPOINT
# =========================
@app.route("/api/quiz", methods=["POST"])
def generate_quiz():
    import json, re
    data = request.get_json()
    subject = data.get("subject", "General Knowledge")
    num_questions = int(data.get("num_questions", 5))

    prompt = f"""
    Generate {num_questions} multiple-choice questions on {subject}.
    Return ONLY valid JSON in this exact format:
    [
      {{
        "question": "What is ...?",
        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
        "answer": "A"
      }},
      ...
    ]
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content.strip()

    # 🛠 Extract JSON only (ignore extra text around it)
    json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not json_match:
        return jsonify({"error": "AI did not return valid JSON", "raw": raw_output}), 500

    json_text = json_match.group(0)

    try:
        quiz_json = json.loads(json_text)
    except Exception as e:
        return jsonify({"error": "JSON parsing failed", "raw": raw_output, "fixed": json_text}), 500

    return jsonify({"quiz": quiz_json})


# =========================
# 📌 UPLOAD & INDEX STUDY MATERIAL
# =========================
@app.route("/api/upload", methods=["POST"])
def upload_material():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    text_chunks = []

    # PDF Handling
    if filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.extend(chunk_text(text))

    # TXT Handling
    elif filename.endswith(".txt"):
        text = file.read().decode("utf-8")
        text_chunks.extend(chunk_text(text))

    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # 🗑️ Reset index before inserting new vectors
    try:
        index.delete(delete_all=True)
        print("✅ Pinecone index cleared before upload.")
    except Exception as e:
        print("⚠️ Could not clear Pinecone index:", e)

    # Embed & upsert into Pinecone
    vectors = []
    for chunk in text_chunks:
        vector = embed_model.encode(chunk).tolist()
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    return jsonify({"status": f"✅ Reset index and uploaded {len(vectors)} chunks to Pinecone"})


# =========================
# 📌 Helper: Chunk text into ~200 words
# =========================
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# =========================
# 📌 FRONTEND ROUTES
# =========================
@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/quiz")
def quiz_page():
    return render_template("quiz.html")


# =========================
# 📌 MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
