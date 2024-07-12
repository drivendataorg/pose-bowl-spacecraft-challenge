#!/usr/bin/env bash

DATA_DIR=/code_execution/data
SUBMISSION_PATH=/code_execution/submission/submission.csv

# call our script (main.py in this case) and tell it where the data and submission live
python main.py $DATA_DIR $SUBMISSION_PATH