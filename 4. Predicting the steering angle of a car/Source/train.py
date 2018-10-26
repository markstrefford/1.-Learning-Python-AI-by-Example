"""
train.py

Train the CNN
"""

import pandas as pd
from model import model, LossHistory, tensor_board, check_point
from data import generators

image_size = (256, 455)

# Prepare data for training, validation and test
columns = ['image_name', 'angle', 'date', 'time']
df = pd.read_csv('./data/data.txt', names=columns, delimiter=' ')

sample_idx = {}
num_samples = len(df)
sample_idx['train'] = [i for i in range(0, num_samples, 2)]
sample_idx['valid'] = [i for i in range(1, num_samples, 4)]
sample_idx['test'] = [i for i in range(3, num_samples, 4)]

# Set up a generator
train_generator = generators.DataGenerator(df.loc[sample_idx['train']], data_dir='./data/data')

# Setup the CNN
history = LossHistory()
cnn = model.cnn()

# Train CNN
cnn.fit_generator(train_generator, epochs=1, steps_per_epoch=1, callbacks=[history, tensor_board, check_point])
print(history.losses)
