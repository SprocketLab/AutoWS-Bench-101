#!/bin/bash

# Download WRENCH datasets
gdown --no-cookies http://drive.google.com/uc?id=19wMFmpoo_0ORhBzB6n16B1nRRX508AnJ
unzip -o datasets.zip
rm datasets.zip 

# Download FWRENCH datasets
gdown --no-cookies https://drive.google.com/uc?id=1SPMQNOdw9FhdoNYr3revPF2yHaIK_M_U
unzip -o FWRENCH_datasets.zip
rm FWRENCH_datasets.zip
mv FWRENCH_datasets/* datasets/
rm -rf FWRENCH_datasets
