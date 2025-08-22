"""Helper library for MBTI questions"""
# pylint: disable=line-too-long
from typing import Final

MBTI_QUESTIONS: Final[list[dict[str, str]]] = [
    # Extraversion (E) vs. Introversion (I)
    {
        "id": "Q1",
        "question": "Do you feel more energized when surrounded by people?",
        "trait": "E",
    },
    {
        "id": "Q2",
        "question": "Do you often find solitude more refreshing than social gatherings?",
        "trait": "I",
    },
    {
        "id": "Q3",
        "question": " When faced with a problem, do you prefer discussing it with others??",
        "trait": "E",
    },
    {
        "id": "Q4",
        "question": "Do you tend to process your thoughts internally before you speak?",
        "trait": "I",
    },
    {
        "id": "Q5",
        "question": "At parties, do you initiate conversations with new people?",
        "trait": "E",
    },
    {
        "id": "Q6",
        "question": "Do you prefer spending weekends quietly at home rather than going out?",
        "trait": "I",
    },
    # Sensing (S) vs. Intuition (N)
    {
        "id": "Q7",
        "question": "Do you focus more on the details and facts of your immediate surroundings?",
        "trait": "S",
    },
    {
        "id": "Q8",
        "question": "Are you more interested in exploring abstract theories and future possibilities?",
        "trait": "N",
    },
    {
        "id": "Q9",
        "question": "In learning something new, do you prefer hands-on experience over theory?",
        "trait": "S",
    },
    {
        "id": "Q10",
        "question": "Do you often think about how actions today will affect the future?",
        "trait": "N",
    },
    {
        "id": "Q11",
        "question": "When planning a vacation, do you prefer having a detailed itinerary?",
        "trait": "S",
    },
    {
        "id": "Q12",
        "question": "Do you enjoy discussing symbolic or metaphorical interpretations of a story?",
        "trait": "N",
    },
    # Thinking (T) vs. Feeling (F)
    {
        "id": "Q13",
        "question": "When making decisions, do you prioritize logic over personal considerations?",
        "trait": "T",
    },
    {
        "id": "Q14",
        "question": "Are your decisions often influenced by how they will affect others emotionally?",
        "trait": "F",
    },
    {
        "id": "Q15",
        "question": "In arguments, do you focus more on being rational than on people's feelings?",
        "trait": "T",
    },
    {
        "id": "Q16",
        "question": "Do you strive to maintain harmony in group settings, even if it means compromising?",
        "trait": "F",
    },
    {
        "id": "Q17",
        "question": "Do you often rely on objective criteria to assess situations?",
        "trait": "T",
    },
    {
        "id": "Q18",
        "question": "When a friend is upset, is your first instinct to offer emotional support rather than solutions?",
        "trait": "F",
    },
    # Judging (J) vs. Perceiving (P)
    {
        "id": "Q19",
        "question": "Do you prefer to have a clear plan and dislike unexpected changes?",
        "trait": "J",
    },
    {
        "id": "Q20",
        "question": "Are you comfortable adapting to new situations as they happen?",
        "trait": "P",
    },
    {
        "id": "Q21",
        "question": "Do you set and stick to deadlines easily?",
        "trait": "J",
    },
    {
        "id": "Q22",
        "question": "Do you enjoy being spontaneous and keeping your options open?",
        "trait": "P",
    },
    {
        "id": "Q23",
        "question": "Do you find satisfaction in completing tasks and finalizing decisions?",
        "trait": "J",
    },
    {
        "id": "Q24",
        "question": "Do you prefer exploring various options before making a decision?",
        "trait": "P",
    },
]
