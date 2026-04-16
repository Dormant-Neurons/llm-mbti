""" 
Script to generate the trait datasets using the trait prompt and ChatGPT
the script fills the trait prompt using the defined personas and their descriptions and utilizes
ChatGPT to generate valid JSON files for each trait.
"""
import json
import os
import ast
from dotenv import load_dotenv

from tqdm import tqdm
from openai import OpenAI

from utils.trait_prompt import trait_prompt
from data.profile_personas import Personas


def main():
    # load the OpenAI API key from the .env file
    load_dotenv()
    client = OpenAI()
    # iterate over all Personas and generate a dataset for each trait where the Persona contains "I"
    for persona in tqdm(Personas, desc="Generating trait datasets", unit=" persona"):
        if "BASELINE" in persona.name:
            continue

        if "_I" in persona.name:
            final_prompt = trait_prompt.replace("{TRAIT}", persona.name.split("_I")[0].lower())
            final_prompt = final_prompt.replace("{trait_instruction}", persona.value)

            # prompt ChatGPT with the final prompt and get the response
            response = client.responses.create(
                model="gpt-5.4",
                input=final_prompt,
            )

            parsed_response = ast.literal_eval(response.output_text)

            # save the response as a JSON file in the data/trait_datasets directory
            output_dir = "data/trait_datasets"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"{persona.name.split("_I")[0].lower()}.json"
            )
            with open(output_path, mode="w", encoding="utf-8") as f:
                json.dump(parsed_response, f, indent=4)

if __name__ == "__main__":
    main()
