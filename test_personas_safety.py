"""Test prototype for evaluating LLMs on the MBTI test."""

# pylint: disable=not-an-iterable
import os
# import json
import argparse
import time
import datetime
import subprocess
import psutil
import getpass

#from langchain_ollama import ChatOllama
from ollama import chat
# from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig
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

    # if device == torch.device("cpu", 0) or "cuda" in str(device):
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type= "nf4"
    #     )
    #     llm = AutoModelForCausalLM.from_pretrained(
    #         model,
    #         device_map="auto",
    #         dtype=torch.bfloat16,
    #         quantization_config=quantization_config
    #     )
    # else:
    #     llm = AutoModelForCausalLM.from_pretrained(
    #         model,
    #         device_map="auto",
    #         dtype=torch.float16,
    #     )
    # tokenizer = AutoTokenizer.from_pretrained(model)

    # iterate over all personalities from the personas definition
    for personality in Personas:
        print(f"{TColors.OKBLUE}Testing personality: {TColors.ENDC}{personality.name}")
        # iterate over all MBTI questions and evaluate the LLMs answers
        answer_list = []
        answer_dict = {} # dict with the answers for every question

        for question in tqdm(
            safety_questions, desc="Evaluating Safety Questions", unit="question"
        ):
            for _ in range(pass_at_k): # 1 attempts per question
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
                # response = Answer(
                #     answer=response["message"]["content"]["answer"],
                #     explanation=response["message"]["content"]["explanation"],
                # )
                response = Answer.model_validate_json(response["message"]["content"])
                if response is not None and response.answer in [0, 1, 2, 3]:
                    answer_dict[question["question_id"]] = response
                    answer_list.append(response)
                    break
            else:
                response = Answer(
                    answer=0,
                    explanation="The model failed to provide a valid response."
                )

                answer_dict[question["question_id"]] = response
                answer_list.append(response)

            #     messages=[
            #         {
            #             "role": "system",
            #             "content": personality.value,
            #         },
            #         {
            #             "role": "user",
            #             "content": question_prefix + hexaco_questions[question_key],
            #         },
            #     ]
            #     messages = tokenizer.apply_chat_template(
            #         messages,
            #         tokenize=False,
            #         add_generation_prompt=True,
            #     )
            #     #response = llm.invoke(messages)
            #     inputs = tokenizer(
            #         messages,
            #         return_tensors="pt",
            #         add_special_tokens=False
            #     ).to(device)
            #     outputs = llm.generate(
            #         **inputs,
            #         pad_token_id=tokenizer.eos_token_id,
            #         #output_hidden_states=True,
            #         max_new_tokens=2048,
            #         #do_sample=True,
            #     )

            #     # extract the model activations of the hidden layers
            #     # TODO

            #     response = tokenizer.batch_decode(
            #         outputs,
            #         skip_special_tokens=True
            #     )[0]

            #     # parse the response to the structured output to json
            #     try:
            #         # remove system prompt from the response
            #         response = response.split("my answer:")[1].strip()

            #         # extract answer and explanation properties from the response
            #         answer = response.split("\"answer\":")[1].split(",")[0].strip().replace("\"", "")
            #         explanation = response.split("\"explanation\":")[1].split(",")[0].strip().replace("\"", "")
            #         response = Answer(
            #            answer=answer,
            #            explanation=explanation
            #         )

            #     except (json.JSONDecodeError, KeyError, IndexError) as e:
            #         print("Failed to parse response:", e)
            #         response = None

            #     if response is not None:
            #         hexaco_dict[question_key] = response
            #         answer_list.append(response)
            #         break
            # else:
            #     response = Answer(
            #         answer="0",
            #         explanation="The model failed to provide a valid response."
            #     )

            #     hexaco_dict[question_key] = response
            #     answer_list.append(response)

        # fix the model specifier for path names, aka. remove "/" characters
        model_str = model.replace("/", "-").replace(":", "-")

        # calculate the total correct answers
        total_correct_answers = 0
        for question in safety_questions:
            if question["question_id"] in answer_dict:
                model_answer = answer_dict[question["question_id"]].answer
                correct_answer = answer_keys[question["question_id"]].index("correct_answer")
                if model_answer == correct_answer:
                    total_correct_answers += 1

        personality_dict[personality.name] = total_correct_answers / len(safety_questions) * 100
        # log the conversation
        log_safety_questions_results(
            llm_type=model_str,
            personality=personality.name,
            log_path="logs/",
            total_questions=len(safety_questions),
            total_correct=total_correct_answers,
        )

    # use matplotlib to create a bar chart of the results
    labels = list(personality_dict.keys())
    values = list(personality_dict.values())
    #x = range(len(labels))
    width = 0.4  # the width of the bars
    _, ax = plt.subplots(figsize=(36, 18))
    # for i, (persona, correct_answers) in enumerate(personality_dict.items()):
    #     safety_values = [correct_answers for _ in labels]
    #     ax.bar([p + i * width for p in x], safety_values, width, label=persona)
    ax.bar(labels, values, width)
    ax.set_xlabel("Personalities")
    ax.set_ylabel("Correct Answers (%)")
    ax.set_title(f"Safety Questions Test Results - {model_str} - pass@{pass_at_k}")
    #ax.set_xticks([p + (len(personality_dict) - 1) * width / 2 for p in x])
    #ax.set_xticklabels(labels)
    ax.legend()
    #plt.tight_layout()
    plt.savefig(f"logs/{model_str}_safety_questions_pass@{pass_at_k}.png")
    plt.show()

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
