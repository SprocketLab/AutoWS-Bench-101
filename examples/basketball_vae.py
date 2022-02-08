import torch
import torch.nn as nn
import numpy as np
# load numpy array from csv file
from numpy import loadtxt
from numpy import savetxt
from wrench.dataset import load_dataset
from scipy.stats import mode
from sklearn import metrics
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from helper_func.helper_utils import set_deterministic, set_all_seeds
from helper_func.helper_train import train_vae_v1

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :2048]


class VAE(nn.Module):
    def __init__(self, z_dim = 10, features = 2048):
        super().__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(features, features),
                nn.Linear(features, z_dim),
                nn.LeakyReLU(negative_slope=0.1)
         )
        
        self.z_mean = torch.nn.Linear(z_dim, z_dim)
        self.z_log_var = torch.nn.Linear(z_dim, z_dim)
        
        self.decoder = nn.Sequential(
                nn.Linear(z_dim, features),
                nn.Linear(features, features),
                nn.ReLU(),
                Trim(),  # 1x29x29 -> 1x28x28
                nn.Sigmoid()
                )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
            
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


if __name__ == "__main__":
    labels = loadtxt('../datasets/basketball/labels.csv', delimiter=',')
    features = loadtxt('../datasets/basketball/features.csv', delimiter=',')
    print(features.shape)
    CUDA_DEVICE_NUM = 1
    #DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cuda')
    print('Device:', DEVICE)

    # Hyperparameters
    RANDOM_SEED = 123
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64
    NUM_EPOCHS = 50

    set_deterministic
    set_all_seeds(RANDOM_SEED)
    ten_feature = torch.Tensor(features)
    ten_label = torch.Tensor(labels)
    my_dataset = TensorDataset(ten_feature, ten_label)
    train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE,
                            drop_last=False,
                            shuffle=False, 
                            num_workers=0)

    print(len(train_loader))
    print(len(my_dataset))
    print(labels[:10])

    set_all_seeds(RANDOM_SEED)

    model = VAE()
    model.to(DEVICE)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)

    log_dict, z_mean, latent = train_vae_v1(beta = 1, num_epochs=NUM_EPOCHS, model=model, 
                        optimizer=optimizer, device=DEVICE, 
                        train_loader=train_loader,
                        skip_epoch_stats=True,
                        logging_interval=50)





