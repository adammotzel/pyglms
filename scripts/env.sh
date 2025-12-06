#!/bin/bash

uv venv --python 3.10

source .venv/Scripts/activate || source .venv/bin/activate
uv sync --all-extras

echo ""
echo "Virtual environment created and activated."
echo ""
