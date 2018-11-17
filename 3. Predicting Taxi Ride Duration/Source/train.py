"""
train.py

Train the NN
"""

import pandas as pd
import argparse
import cv2
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
                    default=200,
                    type=int,
                    help='Number of epochs')
parser.add_argument('-data-file', dest='data-file',
                    default='./data/data.txt',
                    help='File containing list of images and steering angles')
parser.add_argument('-log-images', dest='log-images',
                    default='N',
                    choices=('Y', 'N'),
                    help='Log training images to disk (Y)es or (N)o')
args = vars(parser.parse_args())

# Prepare data for training, validation and test
# TODO - Update columns and handle batch-loading the CSV!! 
columns = ['image_name', 'angle', 'date', 'time']
df = pd.read_csv(args['data-file'], names=columns, delimiter=' ').sample(frac=1).reset_index(drop=True)

# TODO - Determine length of dataset
# TODO - Split into train, validation, test (perhaps indices based on length of dataset?)
sample_idx = {}
num_samples = len(df)
sample_idx['train'] = [i for i in range(0, num_samples, 2)]
sample_idx['valid'] = [i for i in range(1, num_samples, 4)]
sample_idx['test'] = [i for i in range(3, num_samples, 4)]

# Setup debugging
debug = True if args['debug'] == 'Y' else False
if debug:
    print('train.py: batch-size={}, limit-batches={}, epochs={}, data-file={}'
          .format(args['batch-size'], args['limit-batches'], args['epochs'], args['data-file']))

# Set up a generator
train_generator = generators.DataGenerator(df.loc[sample_idx['train']],
                                           debug=debug,
                                           batch_size=args['batch-size'],
                                           limit_batches=args['limit-batches'],
                                           label='Train')
valid_generator = generators.DataGenerator(df.loc[sample_idx['valid']],
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

