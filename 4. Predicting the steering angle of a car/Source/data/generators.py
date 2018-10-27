"""
Data generators for loading training, validation and test data sets

"""

import pandas as pd
import numpy as np
import cv2
import os
from keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Data Generator to load training, validation and test batches
    """
    def __init__(self, df: pd.DataFrame, data_dir='./data', image_size=(256, 455),
                 batch_size=32, debug=False, limit_batches=0):
        """
        :param df:
        :param data_dir:
        :param data_file:
        :param batch_size:
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = 1
        self.idx = 0
        self.batch_count = 0
        self.debug = debug
        self.num_batches = int(np.floor(len(df) / self.batch_size))
        self.limit_batches = limit_batches if limit_batches < self.num_batches and limit_batches else self.num_batches
        self.df = df.reset_index().loc[:self.limit_batches * self.batch_size]
        if debug:
            print('DataGenerator(): num_batches = {}, batch_size = {}, len(df) = {}'
                  .format(self.limit_batches, self.batch_size, len(self.df)))

    def __len__(self):
        """
        Find number of batches per epoch
        """
        return self.limit_batches

    def __getitem__(self, batch_num):
        """
        Generate one batch of data
        :param batch_num:
        :return:
        """
        batch_data = self.df[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        X, y = self.__data_generation(batch_data.reset_index())
        return X, y

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        :param index: list
        """

        X = np.zeros((self.batch_size, *self.image_size, self.channels))
        y = np.zeros((self.batch_size), dtype=float)

        for i, sample in batch_data.iterrows():
            image_path = os.path.join(self.data_dir, sample['image_name'])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            X[i, :, :, 0] = resized
            y[i] = sample['angle']
        return X, y






