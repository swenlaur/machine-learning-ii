from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Constants and hyperparameters for image generation taken from keras documentation
    """
    batch_size: int = 64
    num_channels: int = 1
    num_classes: int = 10
    image_size: int = 28
    latent_dim: int = 128
