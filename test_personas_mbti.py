"""Test prototype for evaluating LLMs on the MBTI test."""
# pylint: disable=not-an-iterable
import os
import argparse
import datetime
import subprocess
import psutil
import getpass
from typing import List

from ollama import chat
import torch
from tqdm import tqdm

from utils.colors import TColors
from utils.personas import PersonalityPrompt
from utils.structures import Answer
from datasets.mbti import MBTI_QUESTIONS



def convert_responses_to_scores(responses: List[Answer]) -> dict[str, int]:
    """
    This function takes responses and calculates the MBTI scores.

    Parameters
    ----------
    responses : list
        A list of responses to the MBTI questions. Each response is of type dict
        and has the following keys: 'text' (str), 'trait' (str), 'answer' (str).

    Returns
    -------
    scores : dict
        A dictionary containing the MBTI scores for each trait.

    """
    # gather a list of answers and convert them to lowercase
    answers_unsanitized = [response.answer.lower() for response in responses]
    # sanitize the answers
    answers = []
    for answer in answers_unsanitized:
        if "yes" in answer:
            answers.append("yes")
        else:
            answers.append("no")

    assert len(answers) == 24, f"{TColors.FAIL}Expected 24 answers, got {len(answers)}"

    # storing the answers in a dichotomy map
    dichotomy_map = {
        "E/I": [1, 0, 1, 0, 1, 0],
        "S/N": [1, 0, 1, 0, 1, 0],
        "T/F": [1, 0, 1, 0, 1, 0],
        "J/P": [1, 0, 1, 0, 1, 0],
    }
    # the final scores are saved in a score dictionary
    scores = {"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0}

    # Iterate through the responses and update scores
    for i, answer in enumerate(answers):
        if i < 6:  # E/I questions
            trait = "E" if answer == "yes" else "I"
            scores[trait] += dichotomy_map["E/I"][i % 6]
        elif i < 12:  # S/N questions
            trait = "S" if answer == "yes" else "N"
            scores[trait] += dichotomy_map["S/N"][i % 6]
        elif i < 18:  # T/F questions
            trait = "T" if answer == "yes" else "F"
            scores[trait] += dichotomy_map["T/F"][i % 6]
        else:  # J/P questions
            trait = "J" if answer == "yes" else "P"
            scores[trait] += dichotomy_map["J/P"][i % 6]

    return scores


def get_mbti_type(scores: dict[str, int]) -> str:
    """
    This function takes the MBTI scores and determines the MBTI type.

    Parameters
    ----------
    scores : dict
        A dictionary containing the MBTI scores for each trait.

    Returns
    -------
    mbti_type : str
        The determined MBTI type as a string.
    """
    # Process the scores to determine MBTI type
    mbti_type = ""
    for trait_pair in ["EI", "SN", "TF", "JP"]:
        trait1, trait2 = trait_pair
        if scores[trait1] >= scores[trait2]:
            mbti_type += trait1
        else:
            mbti_type += trait2
    return mbti_type


def main(device: str, model: str) -> None:
    """
    Main function for testing MBTI personality prompts.

    Parameters:
        device: str - The device to run the model on (e.g., "cpu", "cuda", "mps")
        model: str - The model to use for inference (e.g., "llama3.1")

    Returns:
        None
    """
    # ──────────────────────────── set devices and print informations ─────────────────────────
    # set the devices correctly
    if "cpu" in device:
        device = torch.device("cpu", 0)
    elif "cuda" in device and torch.cuda.is_available():
        if "cuda" not in device.split(":")[-1]:
            device = torch.device("cuda", int(device.split(":")[-1]))
        else:
            device = torch.device("cuda", 0)
    elif "mps" in device and torch.backends.mps.is_available():
        if "mps" not in device.split(":")[-1]:
            device = torch.device("mps", int(device.split(":")[-1]))
        else:
            device = torch.device("mps", 0)
    else:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} "
            f"{TColors.ENDC}is not available. Setting device to CPU instead."
        )
        device = torch.device("cpu", 0)

    # pull the ollama model
    subprocess.call(f"ollama pull {model}", shell=True)

    # have a nice system status print
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 23)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: "
        + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: "
        f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and "
        f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda", 0)) and torch.cuda.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: "
            f"{torch.cuda.mem_get_info()[1] // 1024**2} MB"
        )
    elif (
        device == "mps" or torch.device("mps", 0)
    ) and torch.backends.mps.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 14)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model{TColors.ENDC}: {model}"
    )
    print("#" * os.get_terminal_size().columns + "\n")

    # dict storing the personality types -> "Personality Name": "MBTI Type"
    personality_dict = {}

    #iterate over all personalities from the personas definition
    for personality in PersonalityPrompt:
        print(f"{TColors.OKBLUE}Testing personality: {TColors.ENDC}{personality.name}")
        # iterate over all MBTI questions and evaluate the LLMs answers
        answer_list = []
        for question in tqdm(MBTI_QUESTIONS, desc="Evaluating MBTI Questions", unit=" questions"):
            # add an extra prompt suffix for YES and NO answers
            question_suffix = "Answer with a distinct YES or NO and give an explanation!"

            response = chat(
                messages=[
                    {
                        "role": "system",
                        "content": personality.value,
                    },
                    {
                        "role": "user",
                        "content": question["question"] + question_suffix,
                    },
                ],
                model="llama3.1",
                format=Answer.model_json_schema(),
            )
            answer = Answer.model_validate_json(response["message"]["content"])
            answer_list.append(answer)

        # convert the responses to scores
        scores = convert_responses_to_scores(answer_list)
        # convert score to mbti type
        mbti_type = get_mbti_type(scores)
        personality_dict[personality.name] = mbti_type
        break

    # print the final results
    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Results"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 17)
    )
    for personality, mbti_type in personality_dict.items():
        print(f"{TColors.OKBLUE}Personality: {TColors.ENDC}{personality}, " \
              f"{TColors.OKBLUE}MBTI Type: {TColors.ENDC}{mbti_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBTI Test")
    parser.add_argument(
        "--device",
        "-dx",
        type=str,
        default="cuda",
        help="specifies the device to run the computations on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="llama3.1",
        help="specifies the model to use for inference",
    )
    args = parser.parse_args()
    main(**vars(args))
