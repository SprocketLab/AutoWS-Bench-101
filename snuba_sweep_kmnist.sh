#!/bin/bash

resdir=results/kmnist/snubasweep
mkdir -p ${resdir}
####TODO DO NOT REMOVE THESE RESULTS #rm -rf ${resdir}/* 

for embedding in raw pca resnet18 vae
do
    for em_hard_labels in True False
    do 
        for snuba_cardinality in 1 2
        do 
            for n_labeled_points in 100 500 1000 5000
            do 
                savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
                mkdir -p ${savedir}
                python fwrench/applications/kmnist.py \
                    --embedding=${embedding} \
                    --em_hard_labels=${em_hard_labels} \
                    --snuba_cardinality=${snuba_cardinality} \
                    --n_labeled_points=${n_labeled_points} \
                    |& tee -a ${savedir}/res.log
            done
        done
    done
done

# for seed in 0 1 2 3 4 
# do
#     # Top-1
#     embedding=vae
#     em_hard_labels=False
#     snuba_cardinality=2
#     n_labeled_points=100
#     savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#     # mkdir -p ${savedir}
#     # python examples/fwrench_examples/mnist.py \
#     #     --embedding=${embedding} \
#     #     --lf_class_options=${lf_class_options} \
#     #     --em_hard_labels=${em_hard_labels} \
#     #     --snuba_cardinality=${snuba_cardinality} \
#     #     --n_labeled_points=${n_labeled_points} \
#     #     |& tee -a ${savedir}/res_seed${seed}.log

#     # Top-2
#     embedding=vae
#     em_hard_labels=False
#     snuba_cardinality=2
#     n_labeled_points=500
#     savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#     # mkdir -p ${savedir}
#     # python examples/fwrench_examples/mnist.py \
#     #     --embedding=${embedding} \
#     #     --lf_class_options=${lf_class_options} \
#     #     --em_hard_labels=${em_hard_labels} \
#     #     --snuba_cardinality=${snuba_cardinality} \
#     #     --n_labeled_points=${n_labeled_points} \
#     #     |& tee -a ${savedir}/res_seed${seed}.log


#     # Top-3
#     embedding=vae
#     em_hard_labels=True
#     snuba_cardinality=2
#     n_labeled_points=100
#     savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#     # mkdir -p ${savedir}
#     # python examples/fwrench_examples/mnist.py \
#     #     --embedding=${embedding} \
#     #     --lf_class_options=${lf_class_options} \
#     #     --em_hard_labels=${em_hard_labels} \
#     #     --snuba_cardinality=${snuba_cardinality} \
#     #     --n_labeled_points=${n_labeled_points} \
#     #     |& tee -a ${savedir}/res_seed${seed}.log

#     # Top-4
#     embedding=pca
#     em_hard_labels=True
#     snuba_cardinality=2
#     n_labeled_points=100
#     savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#     # mkdir -p ${savedir}
#     # python examples/fwrench_examples/mnist.py \
#     #     --embedding=${embedding} \
#     #     --lf_class_options=${lf_class_options} \
#     #     --em_hard_labels=${em_hard_labels} \
#     #     --snuba_cardinality=${snuba_cardinality} \
#     #     --n_labeled_points=${n_labeled_points} \
#     #     |& tee -a ${savedir}/res_seed${seed}.log

#     # Top-5
#     embedding=pca
#     em_hard_labels=True
#     snuba_cardinality=2
#     n_labeled_points=500
#     savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#     # mkdir -p ${savedir}
#     # python examples/fwrench_examples/mnist.py \
#     #     --embedding=${embedding} \
#     #     --lf_class_options=${lf_class_options} \
#     #     --em_hard_labels=${em_hard_labels} \
#     #     --snuba_cardinality=${snuba_cardinality} \
#     #     --n_labeled_points=${n_labeled_points} \
#     #     |& tee -a ${savedir}/res_seed${seed}.log

# done

# # Cardinality sweep
# snuba_combo_samples=10_000
# for seed in 0 1 2 3 4 
# do
#     for snuba_cardinality in 4 8 16
#     do
#         # Top-3
#         embedding=vae
#         em_hard_labels=True
#         n_labeled_points=100
#         savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#         mkdir -p ${savedir}
#         python examples/fwrench_examples/mnist.py \
#             --embedding=${embedding} \
#             --lf_class_options=${lf_class_options} \
#             --em_hard_labels=${em_hard_labels} \
#             --snuba_cardinality=${snuba_cardinality} \
#             --n_labeled_points=${n_labeled_points} \
#             --snuba_combo_samples=${snuba_combo_samples} \
#             |& tee ${savedir}/res_seed${seed}_samples${snuba_combo_samples}.log
#     done
# done

# for seed in 0 1 2 3 4 
# do
#     snuba_cardinality=8

#     # Top-3
#     embedding=vae
#     em_hard_labels=True
#     n_labeled_points=100
#     savedir=${resdir}/embedding_${embedding}/lf_class_options_${lf_class_options}/em_hard_labels_${em_hard_labels}/snuba_cardinality_${snuba_cardinality}/n_labeled_points_${n_labeled_points}
#     mkdir -p ${savedir}
#     python examples/fwrench_examples/mnist.py \
#         --embedding=${embedding} \
#         --lf_class_options=${lf_class_options} \
#         --em_hard_labels=${em_hard_labels} \
#         --snuba_cardinality=${snuba_cardinality} \
#         --n_labeled_points=${n_labeled_points} \
#         --snuba_combo_samples=${snuba_combo_samples} \
#         --default_weight=1.0 \
#         --accuracy_weight=1.0 \
#         --balanced_accuracy_weight=1.0 \
#         --precision_weight=1.0 \
#         --recall_weight=1.0 \
#         --matthews_weight=1.0 \
#         --cohen_kappa_weight=1.0 \
#         --jaccard_weight=1.0 \
#         --fbeta_weight=1.0 \
#         |& tee ${savedir}/res_seed${seed}_samples${snuba_combo_samples}_allmetrics.log
# done
