import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

from typing import Tuple
from typing import Optional


def train_binary_classifier(
        model: keras.Model,
        images: tf.Tensor,
        labels: tf.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        collect_telemetry: bool = True,
        test_images: Optional[tf.Tensor] = None,
        test_labels: Optional[tf.Tensor] = None,
        **kwargs) -> Tuple[keras.Model, Optional[pd.DataFrame]]:
    """
    Trains binary classifier on training data and computes some telemetry data.
    By default, collects telemetry only about training data unless test data is also provided.
    Telemetry collection works under assumption that model is compiled with an argument metrics=['accuracy'].
    All additional arguments are directly passed to keras.Model.train_on_batch function.

    Return a trained model and a dataframe of collected telemetry about training.
    The dataframe contains one line per epoch and shows losses and accuracy on training and test datasets.
    """
    steps_per_epoch = len(images) // batch_size
    extra_block = bool(len(images) % batch_size)

    if len(images) != len(labels):
        raise ValueError('Invalid training data: Sample counts of images and labels do not match.')
    if (test_images is None) != (test_images is None):
        raise ValueError('Incomplete test data')

    telemetry = None
    collect_test_telemetry = False
    if collect_telemetry and (test_images is None):
        telemetry = (pd.DataFrame(range(epochs), columns=['epoch'])
                     .assign(training_loss=np.nan, training_accuracy=np.nan)
                     .set_index('epoch'))
    elif collect_telemetry and (test_images is not None):
        if len(test_images) != len(test_labels):
            raise ValueError('Invalid test data: Sample counts of images and labels do not match.')
        collect_test_telemetry = True
        telemetry = (pd.DataFrame(range(epochs), columns=['epoch'])
                     .assign(training_loss=np.nan, training_accuracy=np.nan)
                     .assign(test_loss=np.nan, test_accuracy=np.nan)
                     .set_index('epoch'))

    with tqdm(total=epochs, desc="Epoch Progress", unit="epoch") as pbar:
        for epoch in range(epochs):

            # Training
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                model.train_on_batch(images[start:end], labels[start:end], **kwargs)
            if extra_block:
                start = steps_per_epoch * batch_size
                end = len(images)
                model.train_on_batch(images[start:end], labels[start:end], **kwargs)

            if not collect_telemetry:
                pbar.update(1)
                continue

            # Telemetry
            telemetry.loc[epoch, ['training_loss', 'training_accuracy']] = model.evaluate(images, labels, verbose=0)
            if collect_test_telemetry:
                telemetry.loc[epoch, ['test_loss', 'test_accuracy']] = model.evaluate(test_images, test_labels, verbose=0)
            pbar.update(1)

    return model, telemetry