import keras
from keras import layers


def mnist_logreg_model() -> keras.Model:
    """
    Returns initialised neural network for computing logistic regression on 28 x 28 greyscale images.
    """
    logreg = keras.Sequential(
        [
            layers.InputLayer((28, 28)),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ],
        name="Logistic_regression"
    )
    logreg.build()
    return logreg
