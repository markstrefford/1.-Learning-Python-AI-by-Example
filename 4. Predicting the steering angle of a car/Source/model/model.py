"""
Create the CNN model in Keras

"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import plot_model


def cnn(input_shape=(256, 455, 1),
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        loss='mean_squared_error',
        optimizer='adam',
        pool_size=(2, 2),
        dropout=0.25,
        output=False):

    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(24, kernel_size, padding=padding, input_shape=input_shape))
    model.add(Activation(activation))

    # 2nd conv layer
    model.add(Conv2D(36, kernel_size, padding=padding))
    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))

    # 3rd conv layer
    model.add(Conv2D(48, kernel_size, padding=padding))
    model.add(Activation(activation))

    # 4th conv layer
    model.add(Conv2D(64, kernel_size, padding=padding))
    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))

    model.add(Flatten())

    # 1st Fully Connected Layer
    model.add(Dense(100))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 2nd Fully Connected Layer
    model.add(Dense(50))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 3rd Fully Connected Layer
    model.add(Dense(10))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # Output
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    if output:
        plot_model(model, to_file='./model.png')

    return model
