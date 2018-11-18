"""
Data generators for loading training, validation and test data sets

"""

import pandas as pd
import numpy as np
import os
from keras.utils import Sequence
from datetime import datetime
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
          'PUDate', 'PUDayOfWeek', 'PUTimeHour', 'PUTimeMin',
          'DOLocationLat', 'DOLocationLong',
          'Precipitation'
        ]
        Output = [
          'Duration',
          'Price'
        ]
        :param index: list
        """
        X = np.zeros((self.batch_size, self.num_features), dtype=float)
        y = np.zeros((self.batch_size), dtype=float)

        for i, sample in batch_data.iterrows():
            # Get lat/long of pickup and dropoff locations
            PULocation = self.taxizone_data[self.taxizone_data['LocationID'] == sample['PULocationID']].centroids.values(0)
            PULocationLong, PULocationLat = PULocation.x, PULocation.y
            DOLocation = self.taxizone_data[self.taxizone_data['LocationID'] == sample['DOLocationID']].centroids.values(0)
            DOLocationLong, DOLocationLat = DOLocation.x, DOLocation.y

            # Get month date, day of week and hours/mins for pickup and drop off
            PUDateTime = datetime.strptime(sample.tpep_pickup_datetime, '%Y-%m-%d %H:%M:%S')
            PUDate = PUDateTime.strftime('%Y-%m-%d')
            PUMonthDate = PUDate.split('-')[2]
            PUDayOfWeek = PUDateTime.weekday()
            PUTimeHour, PUTimeMinute = datetime.strptime(
                sample.tpep_pickup_datetime, '%Y-%m-%d %H:%M:%S'
            ).strftime('%H:%M').split(':')

            Precipitation = self.weather_data[self.weather_data['DATE'] == PUDate]['PRCP'].values[0]

            X[i] = [
                PULocationLat,
                PULocationLong,
                PUDayOfWeek,
                PUMonthDate,
                PUTimeHour,
                PUTimeMinute,
                DOLocationLat,
                DOLocationLong,
                Precipitation
            ]

            y[i] = [
                sample['trip_distance'],
                sample['total_amount'] - sample['tip-amount']
            ]

            # Extract relevant columns
            # Add in geo location for PU and DO Location IDs
            # Populate X
            # Populate y with price and duration
            if self.debug:
                print(X[i], y[i])
        return X, y






