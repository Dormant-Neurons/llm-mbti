"""Helper library for structured output definitions."""
from pydantic import BaseModel

class Answer(BaseModel):
    """Represents an structured answer for the model output"""

    answer: str
    explanation: str


answer_json_schema = {
    "title": "Answer",
    "description": "Answer for the personality questions",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The explicit answer to the personality question"
        },
        "explanation": {
            "type": "string",
            "description": "The explanation for the answer"
        }
    },
    "required": ["answer", "explanation"]
}
