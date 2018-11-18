"""
Data generators for loading training, validation and test data sets

"""

import pandas as pd
import numpy as np
import cv2
import os
import scipy.misc
from keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Data Generator to load training, validation and test batches
    """
    def __init__(self, tripdata, weatherdata, batch_size=128, limit_batches=0, label=None, debug=False):
        """
        :param df:
        :param data_dir:
        :param data_file:
        :param batch_size:
        """


        self.batch_size = batch_size
        self.label = label
        self.idx = 0
        self.batch_count = 0
        self.debug = debug
        self.num_batches = int(np.floor(len(tripdata) / self.batch_size))
        self.limit_batches = limit_batches if limit_batches < self.num_batches and limit_batches else self.num_batches
        self.tripdata = tripdata   # tripdata.reset_index().loc[:self.limit_batches * self.batch_size]
        self.weatherdata = weatherdata
        if debug:
            print('DataGenerator(): num_batches = {}, batch_size = {}, len(tripdata) = {}'
                  .format(self.limit_batches, self.batch_size, len(self.tripdata)))

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
        batch_data = self.tripdata[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        X, y = self.__data_generation(batch_data.reset_index())
        return X, y

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        :param index: list
        """
        X = np.zeros((self.batch_size, *self.image_size))
        y = np.zeros((self.batch_size), dtype=float)

        for i, sample in batch_data.iterrows():
            # Extract relevant columns
            # Add in geo location for PU and DO Location IDs
            # Populate X
            # Populate y with price and duration
            if self.log_images:
                # text = 'Frame: {} Angle: {}'.format(i, sample['angle'])
                # cv2.putText(resized, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # file = './logs/images/{}-{}-{}'.format(self.label, i, sample['image_name'])
                # print('Writing debug image to {}'.format(file))
                # cv2.imwrite(file, resized)
        return X, y






