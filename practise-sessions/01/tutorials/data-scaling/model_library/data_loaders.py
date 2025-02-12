import h5py
import keras
import tensorflow as tf

from .image_generation import generate_samples

from typing import Dict
from typing import Tuple


def load_image_pairs(n: int, source_file: str, class_0: str, class_1: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads first n samples of digits for both classes separately from the file, i.e., the outcome contains 2 * n samples.
    Final labels will be zero and one instead of original class labels defined as arguments.
    """
    with h5py.File(source_file, 'r') as hf:
        if class_0 not in hf or class_1 not in hf:
            raise ValueError(f'File {source_file} has invalid format.')
        if n > len(hf[class_0]) or n > len(hf[class_1]):
            raise ValueError(f'File {source_file} does not contain enough samples.')
        if hf[class_0][0].shape != (28, 28) or hf[class_1][0].shape != (28, 28):
            raise ValueError(f'File {source_file} has invalid format: unexpected image sizes.')

        # Tensor magic to get interlaced training set
        images = tf.reshape(tf.stack([hf[class_0][:n], hf[class_1][:n]], axis=1), (-1, 28, 28))
        labels = tf.tile(tf.constant([0, 1], 'int32'), (n,))

    return images, labels


def get_image_pairs(n: int, source: Dict[str, tf.Tensor], class_0: str, class_1: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads first n samples of digits for both classes separately from the file, i.e., the outcome contains 2 * n samples.
    Final labels will be zero and one instead of original class labels defined as arguments.
    """
    if class_0 not in source or class_1 not in source:
        raise ValueError('Source has invalid format.')
    if n > len(source[class_0]) or n > len(source[class_1]):
        raise ValueError('Source does not contain enough samples.')
    if source[class_0][0].shape != (28, 28) or source[class_1][0].shape != (28, 28):
        raise ValueError('Source has invalid format: unexpected image sizes.')

    # Tensor magic to get interlaced training set
    images = tf.reshape(tf.stack([source[class_0][:n], source[class_1][:n]], axis=1), (-1, 28, 28))
    labels = tf.tile(tf.constant([0, 1], 'int32'), (n,))

    return images, labels


def generate_image_pairs(n: int, generator: keras.Model, config,
                         class_0: int, class_1: int, verbose: int = 0) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generates n samples of digits for both classes separately, i.e., the outcome contains 2 * n samples.
    Final labels will be zero and one instead of original class labels defined as arguments.
    """
    images = tf.reshape(tf.stack([
        generate_samples(generator, config, label=class_0, sample_count=n, verbose=verbose),
        generate_samples(generator, config, label=class_1, sample_count=n, verbose=verbose)
    ], axis=1), (-1, 28, 28))
    labels = tf.tile(tf.constant([0, 1], 'int32'), (n,))

    return images, labels