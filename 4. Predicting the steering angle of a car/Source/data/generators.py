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
                 batch_size=128, debug=False):
        """
        :param df:
        :param data_dir:
        :param data_file:
        :param batch_size:
        """

        self.df = df.reindex()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = 1
        self.idx = 0
        self.batch_count = 0
        self.debug = debug

    def __len__(self):
        """
        Find number of batches per epoch
        """
        return int(np.floor(len(self.df) / self.batch_size))

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
            X[i, :, :, 0] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            y[i] = sample['angle']
        return X, y






