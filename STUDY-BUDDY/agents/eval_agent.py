# agents/eval_agent.py
"""
Evaluation Agent
- Grades student answers and provides feedback.
- Returns JSON: {score, feedback, strengths, weaknesses, corrected_answer}
"""
import requests
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_BASE

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = (
    "You are an expert teacher. Grade the student's answer on a 0-100 scale, give concise feedback, "
    "list strengths and weaknesses, and provide a corrected model answer. "
    "Return ONLY a valid JSON object with keys: "
    '{"score": number, "feedback": "...", "strengths": ["..."], "weaknesses": ["..."], "corrected_answer": "..."}'
)

class EvalAgent:
    def __init__(self, model_name: str = GROQ_MODEL):
        self.model = model_name

    def evaluate(self, question: str, reference: str, student_answer: str) -> str:
        user_prompt = (
            f"Question: {question}\n\n"
            f"Reference answer: {reference}\n\n"
            f"Student answer: {student_answer}\n\n"
            "Return ONLY JSON; no extra text."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }

        resp = requests.post(f"{GROQ_API_BASE}/chat/completions", headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
