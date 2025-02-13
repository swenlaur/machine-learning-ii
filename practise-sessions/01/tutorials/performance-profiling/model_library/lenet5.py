import keras
from keras import layers


def mnist_binary_lenet5_model() -> keras.Model:
    """
    Returns initialised neural network for LeNet-5 model that works on 28 x 28 greyscale images.
    Differently form the original model we consider only separation of two digit classes.
    The model structure is adapted from Kaggle webpage LeNet-5 CNN with Keras
    https://www.kaggle.com/code/curiousprogrammer/lenet-5-cnn-with-keras-99-48
    """
    lenet5 = keras.Sequential(
        [
            layers.InputLayer((28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'),
            layers.MaxPool2D(strides=2),
            layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'),
            layers.MaxPool2D(strides=2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ],
        name="LeNet-5"
    )
    lenet5.build()
    return lenet5


