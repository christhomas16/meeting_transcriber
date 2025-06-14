#!/bin/bash

# Create a virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate

pip install -r requirements.txt

python meeting_transcriber.py