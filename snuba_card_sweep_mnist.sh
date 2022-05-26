#!/bin/bash

for seed in 0 1 2
do 

    # Varying number of labels
    resdir=results/mnist/snuba_nlabels
    mkdir -p ${resdir}
    ####TODO DO NOT REMOVE THESE RESULTS #rm -rf ${resdir}/* 

    embedding=raw
    snuba_cardinality=2
    n_labeled_points=100
    em_hard_labels=True
    snuba_combo_samples=1_000

    for n_labeled_points in 100 
    do 
        savedir=${resdir}/n_labeled_points_${n_labeled_points}
        mkdir -p ${savedir}
        python fwrench/applications/mnist.py \
            --embedding=${embedding} \
            --em_hard_labels=${em_hard_labels} \
            --snuba_cardinality=${snuba_cardinality} \
            --n_labeled_points=${n_labeled_points} \
            --snuba_combo_samples=${snuba_combo_samples} \
            |& tee -a ${savedir}/res_seed${seed}.log
    done

    # Varying cardinality
    resdir=results/mnist/snuba_cardinalities
    mkdir -p ${resdir}
    ####TODO DO NOT REMOVE THESE RESULTS #rm -rf ${resdir}/* 

    embedding=raw
    snuba_cardinality=2
    n_labeled_points=100
    em_hard_labels=True
    snuba_combo_samples=1_000

    for snuba_cardinality in 64 128 256 512  #1 2 4 8 16 32 
    do 
        savedir=${resdir}/snuba_cardinality_${snuba_cardinality}
        mkdir -p ${savedir}
        python fwrench/applications/mnist.py \
            --embedding=${embedding} \
            --em_hard_labels=${em_hard_labels} \
            --snuba_cardinality=${snuba_cardinality} \
            --n_labeled_points=${n_labeled_points} \
            --snuba_combo_samples=${snuba_combo_samples} \
            |& tee -a ${savedir}/res_seed${seed}.log
    done
done