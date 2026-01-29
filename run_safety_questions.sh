#!/bin/bash
# run test_personas_safety.py for different parameters

python test_personas_safety.py --model mqd100/Gemma3-Instruct-Abliberated:12b --pass_at_k 1
python test_personas_safety.py --model mqd100/Gemma3-Instruct-Abliberated:12b --pass_at_k 5
python test_personas_safety.py --model mqd100/Gemma3-Instruct-Abliberated:27b --pass_at_k 1
python test_personas_safety.py --model mqd100/Gemma3-Instruct-Abliberated:27b --pass_at_k 5