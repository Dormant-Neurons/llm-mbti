# Watch Your Tone : How Emotions Influence LLMs in Security oder so
Here goes a very nice abstract, DOI stuff and links and maybe citations to the paper and code.


## Setup
1. Create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

2. Add your keys to the `.env` file. They key is necessary to create the steering vectors later on.
```bash
cp .env.template .env
# Then edit the .env file and add your OpenAI key
nano -w .env
```

3. Clone the `persona-vectors` repository into this directory and install their dependencies:
```bash
git clone https://github.com/safety-research/persona_vectors
cd persona_vectors
python -m pip install -r requirements.txt
# remove their trait data, since we will generate our own
rm -rf data_generation/*
cd .. # go back to the main directory
```

## Experiments
This section covers how to run the different experiments in this repository. All logs and figures will be saved in the `logs/` directory.

The default model for all experiments is [`mlabonne/gemma-3-27b-it-abliterated`](https://huggingface.co/mlabonne/gemma-3-27b-it-abliterated).

### Hexaco
Run the Hexaco benchmarkt against a specific model with a variety of personas:
```bash
python test_personas_hexaco.py --model <model_name> --device cuda
```

### Safety questions
The following experiments evaluate a model on a set of safety questions (which can be found [here](/data/safety_dataset.py)). The experiments cover a variety of settings for different personas (defined [here](/data/profile_personas.py)) and emotions (defined [here](/data/emotions.py)) in both system prompt and user inputs. All experiments contain a baseline without a persona or emotion.

The combinations are as follows:

1. The LLM is instructed with a persona in its system prompt, and the user input contains the plain safety questions.
```bash
python test_personas_safety.py \
    --model <model_name> \
    --device cuda \
    --pass_at_k 1 \
    --hierarchy_level system
```

2. The LLM is instructed with a persona in the user input, followed by the plain safety questions.
```bash
python test_personas_safety.py \
    --model <model_name> \
    --device cuda \
    --pass_at_k 1 \
    --hierarchy_level user
```

3. The LLM is instructed neutrally in the system prompt, the user input contains the safety questions with an emotion prefix (e.g. "I am feeling angry. \<safety question\>").
```bash
python test_emotions_safety.py \
    --model <model_name> \
    --device cuda \
    --pass_at_k 1 \
    --hierarchy_level user \
    --question_type prefix
```

4. The LLM is instructed neutrally in the system prompt, the user input contains an emotionalized version of the security questions (which can be found [here](/data/safety_dataset.py) in the `emotionalized_questions` field).
```bash
python test_emotions_safety.py \
    --model <model_name> \
    --device cuda \
    --pass_at_k 1 \
    --hierarchy_level user \
    --question_type emotionalized
```

5. The LLM is instructed with a history of the users previous emotional states (which can be found [here](/data/emotions.py) in the `emotion_history` field) in the system prompt, and the user input contains the plain safety questions.
```bash
python test_emotions_safety.py \
    --model <model_name> \
    --device cuda \
    --pass_at_k 1 \
    --hierarchy_level system
```

### Steering
To apply steering vectors for different personas and emotions, follow the next steps to create the datasets and activation vectors. The `model_str` variable is the model name with all "/" and ":" characters replaced by "-", e.g. `mlabonne-gemma-3-27b-it-abliterated`. The `persona_name` variable is the name of the persona for which you want to create the steering vector, e.g. `evil`.

1. Create the persona datasets 
```bash
python gen_trait_data.py
```
This will create a dataset for each persona in the `persona_vectors/data_generation/` directory.

2. Generate activations using positive and negative system prompts from the previously generated datasets. The files will be saved in the `eval_persona_extract/` directory.
```bash
python -m eval.eval_persona \
    --model <model_name> \
    --trait <persona_name> \
    --output_path eval_persona_extract/<model_str>/<persona_name>_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name <persona_name> \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract
```

3. Generate the steering vectors for the different personas.
```bash
python persona_vectors.generate_vec.py \
    --model_name <model_name> \
    --pos_path eval_persona_extract/<model_str>/<persona_name>_pos_instruct.csv \
    --neg_path eval_persona_extract/<model_str>/<persona_name>_neg_instruct.csv \
    --trait <persona_name> \
    --save_dir persona_vectors/<model_str>/<persona_name>
```

4. Re-run the persona safety question experiments with the `--steering <persona_name>` argument to apply the steering vectors to the model's activations.

### Example for applying steering vectors
This example generates the `evil` persona steering vector for the `mlabonne/gemma-3-27b-it-abliterated` model and uses it for the safety questions experiment. 
```bash
python -m eval.eval_persona \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --trait evil \
    --output_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14  \
    --version extract

python persona_vectors.generate_vec.py \
    --model_name mlabonne/gemma-3-27b-it-abliterated \
    --pos_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/mlabonne-gemma-3-27b-it-abliterated/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/mlabonne-gemma-3-27b-it-abliterated/evil

python test_personas_safety.py \
    --model mlabonne/gemma-3-27b-it-abliterated \
    --device cuda \
    --pass_at_k 1 \
    --hierarchy_level system \
    --steering evil
```