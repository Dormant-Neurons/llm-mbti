"""Test prototype for evaluating LLMs on the MBTI test."""

# pylint: disable=not-an-iterable
import os
# import json
import argparse
import subprocess
import datetime
import psutil
import getpass
from typing import List

from langchain_ollama import ChatOllama
from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.colors import TColors
# from utils.personas import PersonalityPrompt
from utils.profile_personas import Personas
from utils.structures import HexacoAnswer
from utils.logging import log_hexaco_conversation
from data.hexaco import hexaco_questions, reversal, domains_questions

def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Extracts the hidden layer activations from the model during a forward pass.

    Parameters:
        model: AutoModelForCausalLM - The language model from which to extract activations.
        tokenizer: AutoTokenizer - The tokenizer used for preparing inputs.
        text: str - the input text as a string
        device: torch.device - The device on which the model is running ("cpu", "cuda", "mps")

    Returns:
        activations: List[torch.Tensor] - List of tensors representing the hidden layer activations.
    """
    activations = []
    layer_list = [range(model.config.num_hidden_layers + 1)]

    # tokenize text
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    for layer in layer_list:
        activations.append(outputs.hidden_states[layer].squeeze().cpu())

    return activations


def convert_responses_to_scores(hexaco_dict: dict[str, HexacoAnswer]) -> dict[str, float]:
    """
    This function takes responses and calculates the HEXACO scores.

    Parameters:
        responses: list - A list of responses to the HEXACO questions. Each response is of type dict
                          and has the following keys: 'text' (str), 'answer' (str).

    Returns:
        scores: dict - Dictionary containing the HEXACO scores for each domain.

    """
    hexaco_scores = {
        "Honesty": 0,
        "Emotionality": 0,
        "eXtraversion": 0,
        "Agreeableness": 0,
        "Conscientiousness": 0,
        "Openness": 0,
    }

    # iterate over all answers, check the domain and if the answer needs to be reversed
    for key, answer in hexaco_dict.items():
        for domain in domains_questions.keys():
            if key in domains_questions[domain]:
                if key in reversal:
                    # reverse the score: 5 -> 1, 4 -> 2, 3 -> 3, 2 -> 4, 1 -> 5
                    hexaco_scores[domain] += 6 - int(answer.answer)
                else:
                    hexaco_scores[domain] += int(answer.answer)

    # normalize every score to be between 0 and 1
    for key, score in hexaco_scores.items():
        hexaco_scores[key] = score / len(domains_questions[key])

    return hexaco_scores





def main(device: str, model: str) -> None:
    """
    Main function for testing HEXACO personality prompts.

    Parameters:
        device: str - The device to run the model on (e.g., "cpu", "cuda", "mps")
        model: str - The model to use for inference (e.g., "meta-llama/llama-3.1-8B-Instruct")

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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Model{TColors.ENDC}: {model}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Personality Test{TColors.ENDC}: HEXACO")
    print("#" * os.get_terminal_size().columns + "\n")

    # dict storing the personality types -> "Personality Name": "MBTI Type"
    personality_dict = {}

    # define the LLM
    llm = ChatOllama(model=model, temperature=0, device=device)
    llm = llm.with_structured_output(HexacoAnswer)
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
        hexaco_dict = {} # dict with the answers for every question

        for question_key in tqdm(
            hexaco_questions.keys(), desc="Evaluating HEXACO Questions", unit="question"
        ):
            for _ in range(5): # 5 attempts per question
                # add an extra prompt prefix for YES and NO answers
                question_prefix = (
                    """
                    You will be given a statement about yourself and you need to respond with a 
                    distinct score, depending how much you think the statement applies to you. 
                    The scores are:
                    1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree.

                    ONLY answer with the distinct number for your opinion, followed by an 
                    explanation. \n
                    """
                )

                messages=[
                    (
                        "system",
                        personality.value,
                    ),
                    (
                        "human",
                        question_prefix + hexaco_questions[question_key],
                    ),
                ]
                response = llm.invoke(messages)
                if response is not None and response.answer in [1, 2, 3, 4, 5]:
                    hexaco_dict[question_key] = response
                    answer_list.append(response)
                    break
            else:
                response = HexacoAnswer(
                    answer=0,
                    explanation="The model failed to provide a valid response."
                )

                hexaco_dict[question_key] = response
                answer_list.append(response)            

            #     question_prefix = (
            #         """
            #         You will be given a statement about yourself and you need to respond with a 
            #         distinct score, depending how much you think the statement applies to you. 
            #         The scores are:
            #         1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree.

            #         ONLY answer with the distinct number for your opinion, followed by an 
            #         explanation. Use the following JSON format:
                    
            #         ```json
            #         {{
            #             "answer": "<your distinct number>",
            #             "explanation": "<your explanation>"
            #         }}
            #         ```
            #         my answer:
            #         """
            #     )

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

        # convert the responses to scores
        hexaco_scores = convert_responses_to_scores(hexaco_dict)

        # fix the model specifier for path names, aka. remove "/" characters
        model = model.replace("/", "-")

        personality_dict[personality.name] = hexaco_scores
        # log the conversation
        log_hexaco_conversation(
            llm_type=model,
            personality=personality.name,
            questions=hexaco_questions,
            answers=[answer.answer for answer in answer_list],
            explanations=[answer.explanation for answer in answer_list],
            log_path="logs/",
            hexaco_scores=hexaco_scores,
        )

    # use matplotlib to create a bar chart of the results
    labels = list(domains_questions.keys())
    x = range(len(labels))
    width = 0.1  # the width of the bars
    _, ax = plt.subplots(figsize=(36, 18))
    for i, (persona, scores) in enumerate(personality_dict.items()):
        hexaco_values = [scores[domain] for domain in labels]
        ax.bar(
            [p + i * width for p in x],
            hexaco_values,
            width,
            label=persona
        )
    ax.set_xlabel("HEXACO Domains")
    ax.set_ylabel("Scores")
    ax.set_title("HEXACO Personality Test Results")
    ax.set_xticks([p + (len(personality_dict) - 1) * width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()
    #plt.tight_layout()
    plt.savefig(f"logs/{model}_hexaco.png")
    plt.show()

    # print the final results
    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Results"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 17)
    )
    for personality, hexaco_scores in personality_dict.items():
        print(f"{TColors.OKBLUE}Personality: {TColors.ENDC}{personality}")
        print(f"{TColors.OKBLUE}HEXACO Scores: {TColors.ENDC}")
        for domain, score in hexaco_scores.items():
            print(f"     {domain} - Score: {score}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEXACO Test")
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
        default="llama3.1:8b",
        help="specifies the model to use for inference",
    )
    args = parser.parse_args()
    main(**vars(args))
