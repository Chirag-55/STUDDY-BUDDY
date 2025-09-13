# agents/tutor_agent.py
"""
Tutor Agent
- Retrieves context via RAG
- Calls Groq (OpenAI-compatible) to answer with citations summary
"""
import requests
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_BASE
from rag_engine import query

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = (
    "You are a helpful, precise study tutor. Use the provided context to answer the student. "
    "If the context does not contain the answer, say you don't know and offer next steps."
)

class TutorAgent:
    def __init__(self, model_name: str = GROQ_MODEL):
        self.model = model_name

    def answer(self, user_question: str) -> str:
        # 1) Retrieve context from Pinecone
        hits = query(user_question, top_k=4)
        context_texts = [h["text"] for h in hits if h.get("text")]
        context = "\n\n".join(context_texts) if context_texts else "No relevant context found."

        # 2) Build chat request
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question:\n{user_question}\n\nContext:\n{context}"}
            ],
            "temperature": 0.2
        }

        resp = requests.post(f"{GROQ_API_BASE}/chat/completions", headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        # Append short provenance preview
        if context_texts:
            prov = "\n\n-- Sources (top hits) --\n"
            for i, c in enumerate(context_texts[:4], start=1):
                clean = c.replace("\n", " ")
                prov += f"[{i}] {clean[:200]}...\n"
            return text + prov
        return text
