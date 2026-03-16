"""Helper library for MBTI questions"""
# pylint: disable=line-too-long
from typing import Final
from enum import Enum

class Emotions(Enum):
    """Represents the different emotions as prefixes to user questions"""

    HAPPY: Final[str] = ""