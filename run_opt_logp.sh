#!/bin/bash

source activate rdkit
TASK='bootstrap_logp'
HPARAMS='./configs/bootstrap_dqn.json'
CUDA_VISIBLE_DEVICES=0 nohup python \
    -u opt_logp.py \
    -t ${TASK} \
    --hparams ${HPARAMS} &> ${TASK}.out &
