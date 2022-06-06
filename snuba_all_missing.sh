#!/bin/bash

# Snuba sweep over all datasets and embeddings
resdir=results/neurips2022/snuba
mkdir -p ${resdir}

snuba_cardinality=1
n_labeled_points=100
lf_selector=snuba
snuba_iterations=3

emb=raw
dataset=spherical_mnist
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

emb=raw
dataset=ecg
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

emb=resnet18
dataset=ecg
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
