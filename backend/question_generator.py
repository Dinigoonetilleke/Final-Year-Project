import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_question_set(passage, question_types, count=12):
    type_instruction = []

    if "mcq" in question_types:
        type_instruction.append("MCQ questions")
    if "short" in question_types:
        type_instruction.append("short-answer comprehension questions")
    if "true_false" in question_types:
        type_instruction.append("true/false comprehension statements")

    prompt = f"""
You are an English lecturer creating comprehension questions.

Generate a high-quality question set based ONLY on the passage.

Question types required:
{", ".join(type_instruction)}

Rules:
- Do NOT create fill-in-the-blank MCQs.
- MCQs must be meaningful comprehension questions with 4 related options.
- Short-answer questions must test understanding, explanation, inference, or purpose.
- True/False questions must include a mix of true and false statements.
- Answers must be accurate and based only on the passage.
- Include Bloom level: Remember, Understand, Analyze.
- Include difficulty: Easy, Medium, Hard.
- Return exactly 12 questions if possible (mix all selected types).

Return ONLY valid JSON using this format:
{{
  "questions": [
    {{
      "type": "MCQ",
      "difficulty": "Easy",
      "bloom": "Remember",
      "question": "What process powers each star according to the passage?",
      "options": ["Gravitational collapse", "Solar flares", "Nuclear fusion", "Chemical combustion"],
      "answer": "Nuclear fusion"
    }}
  ]
}}

Passage:
{passage}
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        max_output_tokens=1200,
    )

    raw_text = response.output_text.strip()

    try:
        data = json.loads(raw_text)
        return data.get("questions", [])
    except Exception:
        return []