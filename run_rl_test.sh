#!/bin/bash

gamma_list=(0 0.3 0.5 0.8 1)
update_list=(2 5 10 20 40)

for ((i=0;i<5;i++))
do
    cat <<EOF > run_naive_${gamma_list[i]}.sh
#!/bin/bash

source activate rdkit
CUDA_VISIBLE_DEVICES=${i} nohup python -u train_dqn.py -t naive_gamma_${gamma_list[i]} --hparams ./configs/naive_dqn_${gamma_list[i]}.json &> naive_gamma_${gamma_list[i]}.out &
EOF
    cat <<EOF > configs/naive_dqn_${gamma_list[i]}.json
{
  "atom_types": ["C", "O", "N"],
  "max_steps_per_episode": 40,
  "allow_removal": true,
  "allow_no_modification": true,
  "allow_bonds_between_rings": false,
  "allowed_ring_sizes": [5, 6],
  "fingerprint_radius": 3,
  "fingerprint_length": 2048,
  "gamma": ${gamma_list[i]},
  "discount_factor": 0.9,

  "num_episodes": 5000,
  "replay_buffer_size": 5000,
  "batch_size": 128,
  "double_q": true,
  "update_frequency": 20,
  "num_bootstrap_heads": 1,
  "prioritized": false,
  "prioritized_alpha": 0.6,
  "prioritized_beta": 0.4,
  "prioritized_epsilon": 0.000001,
  "save_frequency": 200,
  "max_num_checkpoints": 10,

  "dense_layers": [1024, 512, 128, 32],
  "activation": "ReLU",
  "optimizer": "Adam",
  "batch_norm": true,
  "learning_frequency": 4,
  "learning_rate": 0.0001,
  "learning_rate_decay_steps": 10000,
  "learning_rate_decay_rate": 0.9,
  "adam_beta_1": 0.9,
  "adam_beta_2": 0.999,
  "grad_clipping": 10,

  "gen_number": 100,
  "gen_epsilon": 0.1
}
EOF
done

#for beta_1 in ${beta_1_list[*]}
#do
#    for beta_2 in ${beta_2_list[*]}
#    do
#        cat <<EOF > run_naive_beta_${beta_1}_${beta_2}.sh
##!/bin/bash
#
#source activate rdkit
#TASK='naive_beta_${beta_1}_${beta_2}'
#HPARAMS='./configs/naive_dqn_beta_${beta_1}_${beta_2}.json'
#CUDA_VISIBLE_DEVICES=0 nohup python \
#    -u train_dqn.py \
#    -t ${TASK} \
#    --hparams ${HPARAMS} &> ${TASK}.out &
#EOF
#        cat <<EOF > configs/naive_dqn_beta_${beta_1}_${beta_2}.json
#{
#  "atom_types": ["C", "O", "N"],
#  "max_steps_per_episode": 40,
#  "allow_removal": true,
#  "allow_no_modification": true,
#  "allow_bonds_between_rings": false,
#  "allowed_ring_sizes": [5, 6],
#  "fingerprint_radius": 3,
#  "fingerprint_length": 2048,
#  "gamma": 1.0,
#  "discount_factor": 0.9,
#
#  "num_episodes": 5000,
#  "replay_buffer_size": 5000,
#  "batch_size": 128,
#  "double_q": true,
#  "update_frequency": 20,
#  "num_bootstrap_heads": 1,
#  "prioritized": false,
#  "prioritized_alpha": 0.6,
#  "prioritized_beta": 0.4,
#  "prioritized_epsilon": 0.000001,
#  "save_frequency": 200,
#  "max_num_checkpoints": 10,
#
#  "dense_layers": [1024, 512, 128, 32],
#  "activation": "ReLU",
#  "optimizer": "Adam",
#  "batch_norm": true,
#  "learning_frequency": 4,
#  "learning_rate": 0.0001,
#  "learning_rate_decay_steps": 10000,
#  "learning_rate_decay_rate": 0.9,
#  "adam_beta_1": 0.9,
#  "adam_beta_2": 0.999,
#  "grad_clipping": 10,
#
#  "gen_number": 100,
#  "gen_epsilon": 0.1
#}
#EOF
#    done
#done




