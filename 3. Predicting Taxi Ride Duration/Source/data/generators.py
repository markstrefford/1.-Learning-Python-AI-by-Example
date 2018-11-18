"""
Data generators for loading training, validation and test data sets

"""

import pandas as pd
import numpy as np
import os
from keras.utils import Sequence
import geopandas
from shapely.geometry import Point, LineString


def get_taxi_zones(taxi_zones_file):
    taxi_zones = geopandas.read_file(taxi_zones_file).set_index('OBJECTID')
    zone_ids = taxi_zones.index.tolist()
    taxi_zones['centroids'] = taxi_zones.geometry.centroid
    return taxi_zones, zone_ids


class DataGenerator(Sequence):
    """
    Data Generator to load training, validation and test batches
    """
    def __init__(self, trip_data, weather_data, taxizone_data,
                 num_features = 9, batch_size=128, limit_batches=0,
                 label=None, debug=False):
        """
        :param df:
        :param data_dir:
        :param data_file:
        :param batch_size:
        """
        self.num_features = 9
        self.batch_size = batch_size
        self.label = label
        self.idx = 0
        self.batch_count = 0
        self.debug = debug
        self.num_batches = int(np.floor(len(trip_data) / self.batch_size))
        self.limit_batches = limit_batches if limit_batches < self.num_batches and limit_batches else self.num_batches
        self.trip_data = trip_data   # tripdata.reset_index().loc[:self.limit_batches * self.batch_size]
        self.weather_data = weather_data
        self.taxizone_data = taxizone_data
        if debug:
            print('DataGenerator(): num_batches = {}, batch_size = {}, len(tripdata) = {}'
                  .format(self.limit_batches, self.batch_size, len(self.trip_data)))

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
        batch_data = self.trip_data[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        X, y = self.__data_generation(batch_data.reset_index())
        return X, y

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        Input = [
          'PULocationLat', 'PULocationLong',
          'DOLocationLat', 'DOLocationLong',
          'PUDate', 'PUDayOfWeek',
          'DODate', 'DODayOfWeek',
          'Precipitation'
        ]
        Output = [
          'Duration',
          'Price'
        ]
        :param index: list
        """
        X = np.zeros((self.batch_size, self.num_features))
        y = np.zeros((self.batch_size), dtype=float)

        for i, sample in batch_data.iterrows():
            PULocation = self.taxizone_data[self.taxizone_data['PULocationID'] == sample['PULocationID']].centroids
            DOLocation = self.taxizone_data[self.taxizone_data['PULocationID'] == sample['PULocationID']].centroids

            # Extract relevant columns
            # Add in geo location for PU and DO Location IDs
            # Populate X
            # Populate y with price and duration
            if self.debug:
                # text = 'Frame: {} Angle: {}'.format(i, sample['angle'])
                # cv2.putText(resized, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # file = './logs/images/{}-{}-{}'.format(self.label, i, sample['image_name'])
                # print('Writing debug image to {}'.format(file))
                # cv2.imwrite(file, resized)
        return X, y






