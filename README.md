# AutoWS-Bench-101

## Introduction
**AutoWS-Bench-101** is a framework for evaluating automated WS (AutoWS) techniques in challenging WS settings---a set of diverse application domains on which it has been previously difficult or impossible to apply traditional WS techniques.

## Installation
Install anaconda: Instructions here: https://www.anaconda.com/download/  

Clone the repository:
```
https://github.com/Kaylee0501/FWRENCH.git
cd FWRENCH
```
Create virtual environment:
```
conda env create -f env_new_new.yml
source activate FWRENCH
```
## Datasets
Our benchmark auotomatic download the dataset for you. Please run [`data_settings.py`](https://github.com/Kaylee0501/FWRENCH/blob/main/fwrench/utils/data_settings.py) to download the specific dataset you need.

| Name           | # class       | # train      |# valid       |# test        |
| -------------- | ------------- |------------- |------------- |------------- |
| MNIST          | 10            | 57000        | 3000         | 10000        |
| FashionMNIST   | 10            | 57000        | 3000         | 10000        |
| KMNIST         | 10            | 57000        | 3000         | 10000        |
| CIFAR10        | 10            | 47500        | 2500         | 10000        |
| [SphericalMNIST](https://arxiv.org/abs/1801.10130) | 10            | 57000        | 3000         | 10000        |
| PermutedMNIST  | 10            | 57000        | 3000         | 10000        |
| [ECG](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5978770/)            | 4             | 280269       | 14752        | 33494        |
| [EMBER](https://arxiv.org/abs/1804.04637)          | 2             | 285000       | 15000     | 100000|
| [NavierStokes](https://arxiv.org/abs/2010.08895)   | 2             |   100     | 100 |    1900

## Run the Experiment
To run the whole implementation, we provide a [`pipeline`](https://github.com/Kaylee0501/FWRENCH/blob/main/fwrench/applications/pipeline.py). This pipeline will walk you through a full example of our framework. It allows you to choose the datasets and embeddings, generate a bunch of labeling functions (LFs) with our LF selectors. It then trains a `Snorkel` label model and gives you the accuracy and coverage. 
