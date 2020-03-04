#!/bin/bash

# Make sure to install all of the system libraries required by OpenAI Gym
# and PyTorch. You can find how to install them on:
# https://github.com/openai/gym and https://pytorch.org/

virtualenv init -p python3 .virtual
pip install -r requirements.txt
source .virtual/bin/activate
