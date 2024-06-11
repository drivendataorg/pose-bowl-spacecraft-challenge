#!/usr/bin/env bash
pip install --no-deps openvino-2024.0.0-14509-cp311-cp311-manylinux2014_x86_64.whl
DATA_DIR=../data
SUBMISSION_PATH=/code_execution/submission/submission.csv

# call our script (main.py in this case) and tell it where the data is and
python main.py $DATA_DIR $SUBMISSION_PATH