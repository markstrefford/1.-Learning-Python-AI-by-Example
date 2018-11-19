"""
Create the CNN model in Keras

"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint, ProgbarLogger
from keras.optimizers import Adam
from datetime import datetime

# Callbacks
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                          embeddings_data=None)   #, update_freq='epoch')

checkpoint = ModelCheckpoint(filepath='./logs/weights-{}.hdf5'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                             monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

progressbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

# Optimizer and learning rate
lr = 0.001
adam = Adam(lr=lr)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# Input = [
#           'PULocationLat', 'PULocationLong',
#           'DOLocationLat', 'DOLocationLong',
#           Not using distance...  'TripDistance',
#           'PUDate', 'PUDayOfWeek',
#           'PUHour', 'PUMinute',
#           'Precipitation'
#         ]
#
# Could also add in:
#         [
#           'Temperature', 'WindSpeed', 'SnowDepth', 'Snow'
#         ]
# Output = [
#           'Duration',
#           'Price excl tip'
#          ]
def nn(input_shape=9, output_shape=2,
       activation='elu', loss='mean_squared_error',
       optimizer=adam, dropout=0.25, debug=False):

    print('nn(): Creating NN with parameters:\n')
    print('image_shape={}\noutput_shape={}\ndropout={}\nactivation={}\noptimizer={}\nloss={}'
          .format(input_shape, output_shape, dropout, activation, optimizer, loss))

    model = Sequential()

    # 1st Fully Connected Layer
    model.add(Dense(1024, input_dim=input_shape))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 2nd Fully Connected Layer
    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 3rd Fully Connected Layer
    model.add(Dense(128))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # Output
    model.add(Dense(2))
    model.add(Activation('linear'))

    model.compile(loss=loss, optimizer=optimizer)

    if debug:
        print(model.summary())
        plot_model(model, to_file='./logs/model.png')

    return model
