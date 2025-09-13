# agents/question_agent.py
"""
Question Agent
- Generates MCQ and short-answer questions from retrieved context.
- Returns JSON (list of {question, type, choices?, answer})
"""
import requests
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_BASE
from rag_engine import query

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = (
    "You are a quiz generator. Create a mix of MCQ and short-answer questions based on the context. "
    "Output ONLY a valid JSON array. Each item: "
    '{"question": "...", "type": "mcq"|"short", "choices": ["...","..."] (if mcq), "answer": "..."}'
)

class QuestionAgent:
    def __init__(self, model_name: str = GROQ_MODEL):
        self.model = model_name

    def generate(self, topic: str, count: int = 5) -> str:
        hits = query(topic, top_k=6)
        context = "\n\n".join([h["text"] for h in hits if h.get("text")]) or topic

        user_prompt = (
            f"Generate {count} quiz questions about: {topic}.\n"
            f"Use the context below.\nContext:\n{context}\n\n"
            "Remember: Output ONLY JSON (no explanation)."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3
        }

        resp = requests.post(f"{GROQ_API_BASE}/chat/completions", headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
