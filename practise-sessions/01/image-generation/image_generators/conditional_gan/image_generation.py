import keras
import numpy as np
import tensorflow as tf

from keras import ops


def generate_samples(generator, config, label: int, sample_count: int, **kwargs):
    """
    Generates random elements from the fixed image class.
    Return tensor (sample_count, image_size, image_size, 1) of floats in the range [0, 1].
    That is exactly the same format as MNIST dataset is.
    """
    assert len(generator.input_shape) == 2, "Invalid input shape of the generator model"
    assert generator.input_shape[0] is None, "Invalid input shape of the generator model"
    assert generator.input_shape[1] == config.latent_dim + config.num_classes, "Invalid shape of the generator model"

    generator_noise = keras.random.normal(shape=(sample_count, config.latent_dim))
    label_tensor = ops.cast(keras.utils.to_categorical([label], config.num_classes), "float32")
    label_tensor = tf.tile(label_tensor, [sample_count, 1])
    generator_input = ops.concatenate([generator_noise, label_tensor], 1)

    return tf.squeeze(generator.predict(generator_input, **kwargs), -1)


def generate_spurious_samples(generator, config, label_1: int, label_2: int, prop_vec: np.array, **kwargs):
    """
    Generates random surrogate elements between two image classes.
    Elements in prop_vec show the contribution of the first image class.
    Return tensor (sample_count, image_size, image_size, 1) of floats in the range [0, 1].
    That is exactly the same format as MNIST dataset is.
    """
    assert len(prop_vec.shape) == 1, "Invalid shape of the proportion vector"
    assert len(generator.input_shape) == 2, "Invalid input shape of the generator model"
    assert generator.input_shape[0] is None, "Invalid input shape of the generator model"
    assert generator.input_shape[1] == config.latent_dim + config.num_classes, "Invalid shape of the generator model"

    sample_count = len(prop_vec)
    generator_noise = keras.random.normal(shape=(sample_count, config.latent_dim))
    label_1 = ops.cast(keras.utils.to_categorical([label_1], config.num_classes), "float32")
    label_2 = ops.cast(keras.utils.to_categorical([label_2], config.num_classes), "float32")

    prop_vec = ops.cast(ops.reshape(prop_vec, (len(prop_vec), 1)), "float32")
    label_tensor = label_1 * prop_vec + label_2 * (1 - prop_vec)
    generator_input = ops.concatenate([generator_noise, label_tensor], 1)

    return tf.squeeze(generator.predict(generator_input, **kwargs), -1)