#!/usr/bin/bash

source activate rdkit
TASK=''
HPARAMS=''
CUDA_VISIBLE_DEVICES=0 nohup python -u train_DQN.py -t ${TASK} --hparams ${HPARAMS} &> ${TASK}.out &
