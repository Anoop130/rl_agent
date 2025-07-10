#!/bin/bash
set -e  # Exit immediately if a command fails

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment setup complete!"
