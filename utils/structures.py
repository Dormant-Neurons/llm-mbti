"""Helper library for structured output definitions."""
from pydantic import BaseModel

class Answer(BaseModel):
    """Represents an structured answer for the model output"""

    answer: str
    explanation: str
