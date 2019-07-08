#!/bin/bash

TASK='generation'
JSON_DIR='./configs/generation.json'
MODEL_DIR='./checkpoints/opt_logp_5000.pth'
source activate rdkit
python run_generation.py -u -t ${TASK} -p ${MODEL_DIR} --hparams ${JSON_DIR} &> ${TASK}.out &
