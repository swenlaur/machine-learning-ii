import keras
from keras import layers

from .model_config import ModelConfig

config = ModelConfig()
"""
Default constants and hyperparameters for image generation.
"""


# Calculating the number of input channel for the generator and discriminator
#
# In a regular (unconditional) GAN, we start by sampling noise (of some fixed dimension) from a normal distribution.
# In our case, we also need to account for the class labels. We will have to add the number of classes to the input
# channels of the generator (noise input) as well as the discriminator (generated image input).

generator_in_channels = config.latent_dim + config.num_classes
discriminator_in_channels = config.num_channels + config.num_classes

discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

"""
A default discriminator model taken from keras documentation. 
Image size fixes the first two dimensions of the input tensor.
The third dimension is determined by randomness dimension and label count.
The output is the discriminator is zero-one variable. 
"""

generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

"""
A default generator model taken from keras documentation. 
Input layer is of size (sample_count, latent_dim + num_classes). 
The output is of size (sample_count, image_size, image_size, 1).
"""
