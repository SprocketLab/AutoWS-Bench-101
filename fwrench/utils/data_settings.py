from ctypes import util
from fwrench.datasets.torchvision_dataset import CIFAR10Dataset, FashionMNISTDataset
import fwrench.utils as utils
import numpy as np
from fwrench.datasets import (
    MNISTDataset,
    FashionMNISTDataset,
    KMNISTDataset,
    SphericalDataset,
    CIFAR10Dataset,
    ECG_Time_Series_Dataset,
    EmberDataset,
    
)
from wrench.dataset import load_dataset
from wrench.endmodel import EndClassifierModel
import os, shutil


def get_mnist(
    n_labeled_points, dataset_home, data_dir="MNIST_3000",
):

    train_data = MNISTDataset("train", name="MNIST")
    valid_data = MNISTDataset("valid", name="MNIST")
    test_data = MNISTDataset("test", name="MNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_fashion_mnist(
    n_labeled_points, dataset_home, data_dir="FashionMNIST_3000",
):

    train_data = FashionMNISTDataset("train", name="FashionMNIST")
    valid_data = FashionMNISTDataset("valid", name="FashionMNIST")
    test_data = FashionMNISTDataset("test", name="FashionMNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_kmnist(
    n_labeled_points, dataset_home, data_dir="KMNIST_3000",
):

    train_data = KMNISTDataset("train", name="KMNIST")
    valid_data = KMNISTDataset("valid", name="KMNIST")
    test_data = KMNISTDataset("test", name="KMNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_cifar10(
    n_labeled_points, dataset_home, data_dir="CIFAR10_2500",
):

    train_data = CIFAR10Dataset("train", name="CIFAR10")
    valid_data = CIFAR10Dataset("valid", name="CIFAR10")
    test_data = CIFAR10Dataset("test", name="CIFAR10")
    n_classes = 10
    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize KMNIST data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    # TODO Change to ResNet or s2cnn
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_spherical_mnist(
    n_labeled_points, dataset_home, data_dir="SphericalMNIST_3000",
):

    train_data = SphericalDataset("train", name="SphericalMNIST")
    valid_data = SphericalDataset("valid", name="SphericalMNIST")
    test_data = SphericalDataset("test", name="SphericalMNIST")
    os.remove('ember_dataset_2017_2.tar.bz2')
    shutil.rmtree('ember_2017_2')
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    # TODO Change to ResNet or s2cnn
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_permuted_mnist(
    n_labeled_points, dataset_home, data_dir="MNIST_3000",
):

    train_data = MNISTDataset("train", name="MNIST")
    valid_data = MNISTDataset("valid", name="MNIST")
    test_data = MNISTDataset("test", name="MNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Permute the images
    train_data = utils.row_col_permute(train_data)
    valid_data = utils.row_col_permute(valid_data)
    test_data = utils.row_col_permute(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_ecg(
    n_labeled_points, dataset_home, data_dir="ECG_14752",
):

    train_data = ECG_Time_Series_Dataset("train", name="ECG")
    valid_data = ECG_Time_Series_Dataset("valid", name="ECG")
    test_data = ECG_Time_Series_Dataset("test", name="ECG")
    n_classes = 4

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    # train_data = train_data.create_subset(np.arange(3000))
    # test_data = test_data.create_subset(np.arange(1000))
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_ember_2017(
    # 600000 training / 200000 testing
    # sometimes work,sometimes get killed. Maybe only implement a subset?
    n_labeled_points, dataset_home, data_dir="ember_2017_15000",
):

    train_data = EmberDataset('train', name='ember_2017')
    valid_data = EmberDataset('valid', name='ember_2017')
    test_data = EmberDataset('test', name='ember_2017')

    n_classes = 2

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    # train_data = train_data.create_subset(np.arange(3000))
    # test_data = test_data.create_subset(np.arange(1000))
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model