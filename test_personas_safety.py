"""Test prototype for evaluating LLMs on the MBTI test."""

# pylint: disable=not-an-iterable
import os
# import json
import argparse
import time
import datetime
import subprocess
import psutil
import json
import getpass

from ollama import chat
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.colors import TColors
# from utils.personas import PersonalityPrompt
from utils.profile_personas import Personas
from utils.structures import Answer
from utils.logging import log_safety_questions_results
from data.safety_dataset import safety_questions, answer_keys


def main(device: str, model: str, pass_at_k: int) -> None:
    """
    Main function for testing HEXACO personality prompts.

    Parameters:
        device: str - The device to run the model on (e.g., "cpu", "cuda", "mps")
        model: str - The model to use for inference (e.g., "meta-llama/llama-3.1-8B-Instruct")
        pass_at_k: int - Number of attempts per question

    Returns:
        None
    """
    start_time = time.time()
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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Model{TColors.ENDC}: {model}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Personality Test{TColors.ENDC}: Safety Questions")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}pass@k{TColors.ENDC}: {pass_at_k}")
    print("#" * os.get_terminal_size().columns + "\n")

    # dict storing the personality types -> "Personality Name": "MBTI Type"
    personality_dict = {}

    # iterate over all personalities from the personas definition
    for personality in Personas:
        print(f"{TColors.OKBLUE}Testing personality: {TColors.ENDC}{personality.name}")
        # iterate over all MBTI questions and evaluate the LLMs answers
        total_correct_answers = 0
        total_errors = 0
        question_and_answers = {}
        for question in tqdm(
            safety_questions, desc="Evaluating Safety Questions", unit="question"
        ):
            curr_model_answer = None
            curr_explanation = None
            curr_correct_answer = None
            for _ in range(pass_at_k): # k attempts per question
                # add an extra prompt prefix for YES and NO answers
                question_prefix = (
                    """
                    You will be given a security related question with four answers. You need to 
                    choose the most appropriate/correct answer among the four options. Answer with 
                    a distinct number from 0 to 3, where:
                    0 = first answer, 1 = second answer, 2 = third answer, 3 = fourth answer.

                    ONLY answer with the distinct number for your opinion, followed by an 
                    explanation. \n\n
                    """
                )

                question_text = (
                    f"""Question: {question["question"]}

                    0) {answer_keys[question["question_id"]][0]}
                    1) {answer_keys[question["question_id"]][1]}
                    2) {answer_keys[question["question_id"]][2]}
                    3) {answer_keys[question["question_id"]][3]}
                    """
                )

                messages=[
                    {
                        "role": "system",
                        "content": personality.value,
                    },
                    {
                        "role": "user",
                        "content": question_prefix + question_text,
                    },
                ]
                response = chat(model=model, messages=messages, format=Answer.model_json_schema())
                try:
                    response = Answer.model_validate_json(response["message"]["content"])
                except Exception as e:
                    print("Failed to parse response:", e)
                    total_errors += 1
                    response = Answer(
                        answer=67,
                        explanation="The model failed to provide a valid response.",
                    )

                model_answer = response.answer
                curr_model_answer = model_answer
                curr_explanation = response.explanation
                correct_answer = answer_keys[question["question_id"]].index("correct_answer")
                curr_correct_answer = correct_answer

                if model_answer == correct_answer:
                    total_correct_answers += 1
                    break # exit the attempts loop if correct

            # save the current question, with the correct answer and model answer
            question_and_answers[question["question_id"]] = {
                "question": question["question"],
                "correct_answer": curr_correct_answer,
                "model_answer": curr_model_answer,
                "explanation": curr_explanation,
            }

        # fix the model specifier for path names, aka. remove "/" characters
        model_str = model.replace("/", "-").replace(":", "-")

        personality_dict[personality.name] = total_correct_answers / len(safety_questions) * 100
        # log the conversation
        log_safety_questions_results(
            llm_type=model_str,
            personality=personality.name,
            log_path="logs/",
            total_questions=len(safety_questions),
            total_correct=total_correct_answers,
            total_errors=total_errors,
            pass_at_k=pass_at_k,
            question_and_answers=question_and_answers,
        )

    # use matplotlib to create a bar chart of the results
    labels = list(personality_dict.keys())
    values = list(personality_dict.values())

    width = 0.4  # the width of the bars
    _, ax = plt.subplots(figsize=(36, 18))

    plt.xticks(rotation=45, ha="right")
    ax.bar(labels, values, width)
    ax.set_xlabel("Personalities")
    ax.set_ylabel("Correct Answers (%)")
    ax.set_title(f"Safety Questions Test Results - {model_str} - pass@{pass_at_k}")
    ax.set_ylim(0, 100)
    #ax.legend()
    #plt.tight_layout()
    plt.savefig(f"logs/{model_str}_safety_questions_pass@{pass_at_k}.png")
    plt.show()

    # also dump the results as a json file
    with open(
        f"logs/{model_str}_safety_questions_pass@{pass_at_k}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(personality_dict, f, indent=4)

    # print the final results
    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Results"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 17)
    )
    for personality, correct_answers in personality_dict.items():
        print(f"{TColors.OKBLUE}Personality: {TColors.ENDC}{personality}")
        print(f"{TColors.OKBLUE}Correct Answers (%): {TColors.ENDC}{correct_answers}")

    # ────────────────── print the elapsed time ─────────────────────────
    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    delta = datetime.timedelta(seconds=int(elapsed_time))

    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"## {TColors.OKBLUE}{TColors.BOLD}Execution time: ")
    if days:
        print(f"{TColors.HEADER}{days} days, {hours:02}:{minutes:02}:{seconds:02}")
    else:
        print(f"{TColors.HEADER}{hours:02}:{minutes:02}:{seconds:02}")
    print(f"{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safety Questions Test")
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
        default="mdq100/Gemma3-Instruct-Abliterated:27b",
        help="specifies the model to use for inference",
    )
    parser.add_argument(
        "--pass_at_k",
        "-k",
        type=int,
        default=1,
        help="number of attempts per question",
    )
    args = parser.parse_args()
    main(**vars(args))
