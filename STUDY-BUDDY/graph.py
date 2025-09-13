# graph.py
from agents.tutor_agent import TutorAgent
from agents.question_agent import QuestionAgent
from agents.eval_agent import EvalAgent

"""
This file defines a simplified LangGraph-like workflow
that orchestrates the Tutor → Question → Evaluation steps.
"""

def run_workflow(question_text: str, gen_count: int = 5):
    tutor = TutorAgent()
    q_agent = QuestionAgent()
    eval_agent = EvalAgent()

    # 1. Tutor answers student question
    tutor_response = tutor.answer(question_text)

    # 2. Question agent generates a quiz about the same topic
    quiz_questions = q_agent.generate(question_text, count=gen_count)

    # 3. Package result (evaluation is invoked later by API per question)
    return {
        "tutor_answer": tutor_response,
        "quiz_payload": quiz_questions
    }


def evaluate_answer(question: str, student_answer: str):
    """
    Helper to call evaluation agent directly
    """
    eval_agent = EvalAgent()
    feedback = eval_agent.evaluate(question, student_answer)
    return feedback
