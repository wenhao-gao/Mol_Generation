#!/usr/bin/bash

source activate rdkit
TASK=''
HPARAMS=''
CUDA_VISIBLE_DEVICES=0 nohup python -u train_DDQN.py -t ${TASK} --hparams ${HPARAMS} &> ${TASK}.out &
