#!/bin/bash

# Snuba sweep over all datasets and embeddings
resdir=results/neurips2022/clip_zero_shot
mkdir -p ${resdir}

snuba_cardinality=1
n_labeled_points=100
lf_selector=clip_zero_shot
snuba_iterations=3

for seed in 0
do 
    for emb in openai
    do 
        for dataset in permuted_mnist navier_stokes mnist cifar10 spherical_mnist  navier_stokes
        do

            savedir=${resdir}/${seed}/${emb}/${dataset}
            mkdir -p ${savedir}

            CUDA_VISIBLE_DEVICES=1 python -u fwrench/applications/pipeline.py \
                --lf_selector=${lf_selector} \
                --n_labeled_points=${n_labeled_points} \
                --snuba_cardinality=${snuba_cardinality} \
                --snuba_iterations=${snuba_iterations} \
                --seed=${seed} \
                --embedding=${emb} \
                --dataset=${dataset} \
                |& tee -a ${savedir}/res_seed${seed}.log
        done
    done
done
