from matplotlib.pyplot import bar
import numpy as np
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning import LightningModule, Trainer

from .base_embedding import BaseEmbedding
from.vae_utils import ConvVAE, VAE


class VAE2DEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = None

    def fit(self, *data, ngpus=1, max_epochs=5):
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
            hidden_size=128,
            alpha=1024,
            batch_size=144, 
            #save_images=True,
            #save_path='log_images',
            #model_type='vae',
            )
        pl.Trainer(
            gpus=ngpus, max_epochs=max_epochs).fit(
            self.model, datamodule=dm)

    def transform(self, data):
        X_np = self._unpack_data(data, flatten=False)
        X_np_emb, _, = self.model.encode(X_np)

        print()
        print()

        #X_np_emb = X_np.reshape(X_np.shape[0], -1)
        return self._repack_data(data, X_np_emb)

    def fit_transform(self, data):
        return 
        X_np = self._unpack_data(data)
        X_np_emb = X_np.reshape(X_np.shape[0], -1)
        return self._repack_data(data, X_np_emb)
