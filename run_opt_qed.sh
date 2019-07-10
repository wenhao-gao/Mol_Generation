#!/bin/bash

source activate rdkit
TASK='bootstrap_qed'
HPARAMS='./configs/bootstrap_dqn.json'
CUDA_VISIBLE_DEVICES=0 nohup python \
    -u opt_qed.py \
    -t ${TASK} \
    --hparams ${HPARAMS} &> ${TASK}.out &
