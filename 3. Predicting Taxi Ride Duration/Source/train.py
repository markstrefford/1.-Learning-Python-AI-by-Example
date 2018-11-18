"""
train.py

Train the NN
"""

import pandas as pd
import argparse
import cv2
from sklearn.model_selection import train_test_split
from model.model import nn, LossHistory, tensorboard, checkpoint, progressbar
from data import generators


# Process command line arguments if supplied
parser = argparse.ArgumentParser(
        description='Train our cnn to predict steering angles')
show_default = ' (default %(default)s)'

parser.add_argument('-debug', dest='debug',
                    default='N',
                    choices=('Y', 'N'),
                    help='Debug (Y)es or (N)o')
parser.add_argument('-batch-size', dest='batch-size',
                    default=128,
                    type=int,
                    help='Batch size (suggest 32 - 128)')
parser.add_argument('-limit-batches', dest='limit-batches',
                    default=0,
                    type=int,
                    help='Limit batches to train on ')
parser.add_argument('-epochs', dest='epochs',
                    default=20,
                    type=int,
                    help='Number of epochs')
parser.add_argument('-tripdata', dest='tripdata',
                    default='./data/taxi_data/cleansed_yellow_tripdata_2018-06.csv',
                    help='File containing trip data from NYC open data website')
parser.add_argument('-weatherdata', dest='weatherdata',
                    default='./data/weather_data/ny_jfk_weather_2018-06.csv',
                    help='File containing weather data from NYC open data website')
args = vars(parser.parse_args())

# Prepare data for training, validation and test
# Start by loading the data and randomising the order
tripdata = pd.read_csv(args['tripdata'], delimiter=',').sample(frac=1).reset_index(drop=True)
sample_idx = {}
num_samples = len(tripdata)
num_train_samples = int(num_samples * 0.6)
num_valid_samples = int(num_samples * 0.2)
num_test_samples = int(num_samples * 0.2)
sample_idx['train'] = [i for i in range(0, num_train_samples, 1)]
sample_idx['valid'] = [i for i in range(num_train_samples, num_train_samples + num_valid_samples, 1)]
sample_idx['test'] = [i for i in range(num_samples - num_test_samples, num_samples, 1)]

# Load weather data
weatherdata = pd.read_csv(args['weatherdata'], delimiter=',')

# Setup debugging
debug = True if args['debug'] == 'Y' else False
if debug:
    print('train.py: batch-size={}, limit-batches={}, epochs={}, data-file={}'
          .format(args['batch-size'], args['limit-batches'], args['epochs'], args['data-file']))

# Set up a generator
train_generator = generators.DataGenerator(tripdata.loc[sample_idx['train']],
                                           debug=debug,
                                           batch_size=args['batch-size'],
                                           limit_batches=args['limit-batches'],
                                           label='Train')
valid_generator = generators.DataGenerator(tripdata.loc[sample_idx['valid']],
                                           debug=debug,
                                           batch_size=args['batch-size'],
                                           limit_batches=args['limit-batches'],
                                           label='Validate')

# Setup the CNN
history = LossHistory()
nn = nn(debug=debug)

# Train CNN
nn.fit_generator(train_generator, validation_data=valid_generator, epochs=args['epochs'],
                  callbacks=[history, tensorboard, checkpoint, progressbar])
if debug:
    print(history.losses)

