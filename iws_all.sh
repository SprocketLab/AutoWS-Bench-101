#!/bin/bash

# Snuba sweep over all datasets and embeddings
resdir=results/neurips2022/iws
mkdir -p ${resdir}

iws_cardinality=1
n_labeled_points=100
lf_selector=iws
iws_iterations=20

for seed in 0
do 
    for emb in openai
    do 
        for dataset in mnist cifar10 spherical_mnist permuted_mnist ecg ember
        do
        # TODO add navier stokes

            savedir=${resdir}/${seed}/${emb}/${dataset}
            mkdir -p ${savedir}

            python fwrench/applications/pipeline.py \
                --lf_selector=${lf_selector} \
                --n_labeled_points=${n_labeled_points} \
                --iws_cardinality=${iws_cardinality} \
                --iws_iterations=${iws_iterations} \
                --seed=${seed} \
                --embedding=${emb} \
                --dataset=${dataset} \
                |& tee -a ${savedir}/res_seed${seed}.log

        done
    done
done
