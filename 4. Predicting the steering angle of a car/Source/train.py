"""
train.py

Train the CNN
"""

import pandas as pd
import argparse
import cv2
from model.model import cnn, LossHistory, tensorboard, checkpoint, progressbar
from data import generators

image_size = (78, 228)   # (128, 228)  # (256, 455)

# Prepare data for training, validation and test
columns = ['image_name', 'angle', 'date', 'time']
df = pd.read_csv('./data/data.txt', names=columns, delimiter=' ')

sample_idx = {}
num_samples = len(df)
sample_idx['train'] = [i for i in range(0, num_samples, 2)]
sample_idx['valid'] = [i for i in range(1, num_samples, 4)]
sample_idx['test'] = [i for i in range(3, num_samples, 4)]

# Process command line arguments if supplied
parser = argparse.ArgumentParser(
        description='Train our cnn to predict steering angles')
show_default = ' (default %(default)s)'

parser.add_argument('-debug', dest='debug',
                    default='N',
                    choices=('Y', 'N'),
                    help='Debug (Y)es or (N)o')
parser.add_argument('-limit-batches', dest='limit-batches',
                    default=0,
                    type=int,
                    help='Limit batches to train on ')
parser.add_argument('-epochs', dest='epochs',
                    default=10,
                    type=int,
                    help='Number of epochs')

args = vars(parser.parse_args())

# Setup debugging
debug = True if args['debug'] == 'Y' else False
if debug:
    print('train.py: limit-batches={}, epochs={}'.format(args['limit-batches'], args['epochs']))
    cv2.startWindowThread()

# Set up a generator
train_generator = generators.DataGenerator(df.loc[sample_idx['train']],
                                           data_dir='./data/data',
                                           image_size=image_size,
                                           debug=debug,
                                           limit_batches=args['limit-batches'],
                                           label='Train')
valid_generator = generators.DataGenerator(df.loc[sample_idx['valid']],
                                           data_dir='./data/data',
                                           image_size=image_size,
                                           debug=debug,
                                           limit_batches=args['limit-batches'],
                                           label='Validate')

# Setup the CNN
history = LossHistory()
cnn = cnn(input_shape=(*image_size, 1), debug=debug)

# Train CNN
cnn.fit_generator(train_generator, validation_data=valid_generator, epochs=args['epochs'],
                  callbacks=[history, tensorboard, checkpoint, progressbar])
if debug:
    print(history.losses)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
