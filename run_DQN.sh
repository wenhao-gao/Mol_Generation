#!/bin/bash

source activate rdkit
TASK='test'
HPARAMS='./configs/naive_dqn.json'
CUDA_VISIBLE_DEVICES=0 nohup python \
    -u train_dqn.py \
    -t ${TASK} \
    --hparams ${HPARAMS} &> ${TASK}.out &
