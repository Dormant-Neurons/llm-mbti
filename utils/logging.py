"""Helper library for logging the LLMs answers."""
# pylint: disable=inconsistent-quotes
import os
import datetime
from typing import Optional

def log_mbti_conversation(
    llm_type: str,
    personality: str,
    questions: list[str],
    answers: list[str],
    explanations: list[str],
    log_path: str,
    mbti_type: str,
    overwrite: Optional[bool] = True,
) -> None:
    """
    Logs the questions for the MBTI test and the answers with explanations.

    Parameters:
        llm_type: str - The type of LLM used
        personality: str - The personality being tested
        questions: list[str] - The list of questions asked
        answers: list[str] - The list of answers given by the LLM
        explanations: list[str] - The list of explanations provided by the LLM
        log_path: str - The path to the directory where logs should be saved
        mbti_type: str - The MBTI type determined from the answers
        overwrite: Optional[bool] - Whether to overwrite existing log files (default: True)

    Returns:
        None
    """
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if "/" in llm_type:
        llm_type = llm_type.replace("/", "-")

    file_name = log_path + f"{llm_type}_{personality}_mbti_logs.txt"

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file=file_name, mode=mode, encoding="utf-8") as f:
        f.write("\n" + "#" * 100)
        f.write(
            f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n"
        )
        f.write(f">>LLM Type: {llm_type}\n")
        f.write(f">>Personality: {personality}\n")
        f.write(f">>MBTI Type: {mbti_type}\n\n")

        for question, answer, expl in zip(questions, answers, explanations):
            f.write(f">>Question {question['id']}: {question['question']}\n")
            f.write(f">>Trait: {question['trait']}\n")
            f.write(f">>Answer: {answer}\n")
            f.write(f">>Explanation: {expl}\n\n")
        f.write("\n" + "#" * 100)

def log_hexaco_conversation(
    llm_type: str,
    personality: str,
    questions: list[str],
    answers: list[str],
    explanations: list[str],
    log_path: str,
    hexaco_scores: dict[str, float],
    overwrite: Optional[bool] = True,
) -> None:
    """
    Logs the questions for the HEXACO test and the answers with explanations.

    Parameters:
        llm_type: str - The type of LLM used
        personality: str - The personality being tested
        questions: list[str] - The list of questions asked
        answers: list[str] - The list of answers given by the LLM
        explanations: list[str] - The list of explanations provided by the LLM
        log_path: str - The path to the directory where logs should be saved
        hexaco_scores: dict[str, float] - The HEXACO scores determined from the answers
        overwrite: Optional[bool] - Whether to overwrite existing log files (default: True)

    Returns:
        None
    """
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if "/" in llm_type:
        llm_type = llm_type.replace("/", "-")

    file_name = log_path + f"{llm_type}_{personality}_hexaco_logs.txt"

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file=file_name, mode=mode, encoding="utf-8") as f:
        f.write("\n" + "#" * 100)
        f.write(
            f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n"
        )
        f.write(f">>LLM Type: {llm_type}\n")
        f.write(f">>Personality: {personality}\n")
        f.write(">>HEXACO Scores:\n")
        for domain, score in hexaco_scores.items():
            f.write(f"     {domain} - Score: {score}\n")

        f.write("\n")
        for question, answer, expl in zip(questions.keys(), answers, explanations):
            f.write(f">>Question {question}: {questions[question]}\n")
            f.write(f">>Answer: {answer}\n")
            f.write(f">>Explanation: {expl}\n\n")
        f.write("\n" + "#" * 100)


def log_safetybench_results(
    llm_type: str,
    personality: str,
    log_path: str,
    total_questions: int,
    total_correct: int,
    topic_dict: dict[str, int],
    topic_length_dict: dict[str, int],
    overwrite: Optional[bool] = True,
) -> None:
    """
    Logs the questions for the HEXACO test and the answers with explanations.

    Parameters:
        llm_type: str - The type of LLM used
        personality: str - The personality being tested
        log_path: str - The path to the directory where logs should be saved
        total_questions: int - The total number of questions asked
        total_correct: int - The total number of correct answers given by the LLM
        topic_dict: dict[str, int] - The dict with topics and their correct counts
        topic_length_dict: dict[str, int] - The dict with topics and their total question counts
        overwrite: Optional[bool] - Whether to overwrite existing log files (default: True)

    Returns:
        None
    """
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if "/" in llm_type:
        llm_type = llm_type.replace("/", "-")

    file_name = log_path + f"{llm_type}_{personality}_safetybench_logs.txt"

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file=file_name, mode=mode, encoding="utf-8") as f:
        f.write("\n" + "#" * 100)
        f.write(
            f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n"
        )
        f.write(f">>LLM Type: {llm_type}\n")
        f.write(f">>Personality: {personality}\n")
        f.write(f">>Total Questions: {total_questions}\n")
        f.write(f">>Total Correct: {total_correct}\n")
        f.write(f">>Total Accuracy: {total_correct / total_questions * 100:.4f}%\n\n")
        f.write(">>Topic-wise Accuracy:\n")
        for topic, correct in topic_dict.items():
            length = topic_length_dict[topic]
            f.write(
                f"     {topic} - {correct}/{length} = {correct / length * 100:.4f}%\n"
            )
        f.write("\n" + "#" * 100)


def log_safety_questions_results(
    llm_type: str,
    personality: str,
    log_path: str,
    total_questions: int,
    total_correct: int,
    overwrite: Optional[bool] = True,
) -> None:
    """
    Logs the questions for the HEXACO test and the answers with explanations.

    Parameters:
        llm_type: str - The type of LLM used
        personality: str - The personality being tested
        log_path: str - The path to the directory where logs should be saved
        total_questions: int - The total number of questions asked
        total_correct: int - The total number of correct answers given by the LLM
        overwrite: Optional[bool] - Whether to overwrite existing log files (default: True)

    Returns:
        None
    """
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if "/" in llm_type:
        llm_type = llm_type.replace("/", "-")

    file_name = log_path + f"{llm_type}_{personality}_safetybench_logs.txt"

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file=file_name, mode=mode, encoding="utf-8") as f:
        f.write("\n" + "#" * 100)
        f.write(
            f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n"
        )
        f.write(f">>LLM Type: {llm_type}\n")
        f.write(f">>Personality: {personality}\n")
        f.write(f">>Total Questions: {total_questions}\n")
        f.write(f">>Total Correct: {total_correct}\n")
        f.write(f">>Total Accuracy: {total_correct / total_questions * 100:.4f}%\n\n")
        f.write(">>Topic-wise Accuracy:\n")
        f.write("\n" + "#" * 100)