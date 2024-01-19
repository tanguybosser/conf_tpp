#!/bin/bash

models=(
    'gru_cond-log-normal-mixture_temporal_with_labels_conformal'
    'poisson_gru_rmtpp_temporal_with_labels_conformal'
    'poisson_gru_thp_temporal_with_labels_conformal'
    'poisson_gru_sahp_temporal_with_labels_conformal'
    'poisson_gru_mlp-cm_learnable_with_labels_conformal'
    'identity_poisson_times_only'
)

datasets=(
    'lastfm_filtered'
    'mooc_filtered'
    'reddit_filtered_short'
    'retweets_filtered_short'
    'stack_overflow_filtered'
)

conformalizers=(
    'C-Const'
    'C-QRL'
    'N-QRL'
    'C-QR'
    'N-QR'
    'N-HDR-T'
    'C-HDR-T'
    'C-RAPS'
    'N-RAPS'
    'C-APS'
    'N-APS'
    'C-PROB'
    'C-QRL-RAPS'
    'N-QRL-RAPS'
    'C-HDR-RAPS'
    'N-HDR-RAPS'
    'C-HDR'
    'N-HDR'
)


for model in ${models[@]}; do
    for dataset in ${datasets[@]}; do
        for split in {0..4}; do
            python -m conformal.run --no-mlflow --dataset $dataset --load-from-dir 'data/baseline3' \
            --model-name "${dataset}_${model}_split${split}" \
            --model-name-short "${model}" \
            --save-check-dir "checkpoints/full/${dataset}" \
            --save-results-dir "results/full/${dataset}" \
            --batch-size 4 --split $split \
            --alphas-cal 0.2 \
            --delta-end 10 \
            --lambda-reg 0.01 \
            --k-reg 3 \
            --conformalizers ${conformalizers[@]} \
            --exp-dir "final" \
            --n-partitions 4
        done
    done
done


for model in ${models[0]}; do # Here we only run the CLNM model
    for dataset in ${datasets[@]}; do
        for split in {0..4}; do
            python3 -m conformal.run --no-mlflow --dataset $dataset --load-from-dir 'data/baseline3' \
            --model-name "${dataset}_${model}_split${split}" \
            --model-name-short "${model}" \
            --save-check-dir "checkpoints/full/${dataset}" \
            --save-results-dir "results/full/${dataset}" \
            --batch-size 4 --split $split \
            --alphas-cal 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
            --delta-end 10 \
            --lambda-reg 0.01 \
            --k-reg 3 \
            --conformalizers ${conformalizers[@]} \
            --exp-dir "final" \
            --eval-cond-coverage "False" # It is faster to not evaluate conditional coverage
        done
    done
done
