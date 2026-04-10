"""Helper library for structured output definitions."""
from pydantic import BaseModel, Field

class Answer(BaseModel):
    """Represents an structured answer for the model output"""

    answer: int = Field(description="The distinct number to answer the question.")
    explanation: str = Field(description="The explanation for the answer.")

class HexacoAnswer(BaseModel):
    """Represents an structured answer for the model output"""

    answer: int = Field(description="The distinct number to answer the question.")
    explanation: str = Field(description="The explanation for the answer.")

class SafetyBenchAnswer(BaseModel):
    """Represents an structured answer for the model output"""

    answer: str = Field(description="The distinct answer to the question.")


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
