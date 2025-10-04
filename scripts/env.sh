#!/bin/bash

uv venv venv --python 3.12

if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Could not find the activation script for the virtual environment."
fi

uv pip install -r requirements-dev.txt

echo ""
echo "Virtual environment created and activated."
echo ""
