from .vae import VAE, Flatten, Stack  # noqa: F401
from .conv_vae import ConvVAE  # noqa: F401

__all__ = [
    'VAE', 'Flatten', 'Stack'
    'ConvVAE',
]
vae_models = {
    "conv-vae": ConvVAE,
    "vae": VAE
}
