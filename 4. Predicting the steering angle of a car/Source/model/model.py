"""
Create the CNN model in Keras

"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint, ProgbarLogger


# Callbacks
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                          embeddings_data=None)   #, update_freq='epoch')

checkpoint = ModelCheckpoint(filepath='./logs/weights.hdf5', monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)

progressbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def cnn(input_shape=(256, 455, 1),
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        loss='mean_squared_error',
        optimizer='adam',
        pool_size=(2, 2),
        dropout=0.25,
        debug=False):

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
    # model.add(Dropout(dropout))

    # 2nd Fully Connected Layer
    model.add(Dense(50))
    model.add(Activation(activation))
    # model.add(Dropout(dropout))

    # 3rd Fully Connected Layer
    model.add(Dense(10))
    model.add(Activation(activation))
    # model.add(Dropout(dropout))

    # Output
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    if debug:
        plot_model(model, to_file='./model.png')

    return model
