#!/bin/bash

# Snuba sweep over all datasets and embeddings
resdir=results/neurips2022/snuba_cardinality
mkdir -p ${resdir}

snuba_cardinality=1
n_labeled_points=100
lf_selector=snuba
snuba_iterations=3

for seed in 0
do 
    for emb in openai
    do 
        for dataset in mnist cifar10 spherical_mnist permuted_mnist navier_stokes
        do
            for snuba_cardinality in 2 4 8 10
            do

                savedir=${resdir}/${seed}/${emb}/${dataset}/${snuba_cardinality}
                mkdir -p ${savedir}

                CUDA_VISIBLE_DEVICES=2 python -u fwrench/applications/pipeline.py \
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
done
