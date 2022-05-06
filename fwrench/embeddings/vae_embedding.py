from matplotlib.pyplot import bar
import numpy as np
import torch 
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning import LightningModule, Trainer

from .base_embedding import BaseEmbedding
from.vae_utils import ConvVAE, VAE


class VAE2DEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = None

    def fit(self, *data, ngpus=1, max_epochs=20, hidden_size=128): # TODO
        X_nps = []
        for d in data:
            X_nps.append(self._unpack_data(d, flatten=False))
        X_np = np.concatenate(X_nps)

        dm = SklearnDataModule(X_np, X_np)
        self.model = VAE(
            channels=1,
            height=X_np.shape[-2],
            width=X_np.shape[-1], 
            lr=0.005,
            hidden_size=hidden_size,
            alpha=1024,
            batch_size=144, 
            #save_images=True,
            #save_path='log_images',
            #model_type='vae',
            ).cuda()
        pl.Trainer(
            gpus=ngpus, max_epochs=max_epochs).fit(
            self.model, datamodule=dm)
        self.model.eval()

    def transform(self, data):
        X_np = self._unpack_data(data, flatten=False)
        with torch.no_grad():
            X = torch.from_numpy(X_np)
            X = X.type(torch.FloatTensor)
            X.cuda()
            X_emb, _, = self.model.encode(X)
            X_emb = X_emb.cpu().numpy()
        return self._repack_data(data, X_emb)

    def fit_transform(self, data, ngpus=1, max_epochs=5, hidden_size=128):
        self.fit(data, 
            n_gpus=ngpus, 
            max_epochs=max_epochs, 
            hidden_size=hidden_size)
        return self.transform(data)
