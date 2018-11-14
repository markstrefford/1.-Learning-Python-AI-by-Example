#!/usr/bin/env bash

#Open taxi data from http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml

#Taxi zone data
wget https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv -P ./geo_data
wget https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip
unzip ./taxi_zones.zip

#Yellow cab data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2017-07.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2017-08.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2017-09.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2017-10.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2017-11.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2017-12.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-01.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-02.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-03.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-04.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-05.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-06.csv -P ./taxi_data

#Green cab data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2017-07.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2017-08.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2017-09.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2017-10.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2017-11.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2017-12.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2018-01.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2018-02.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2018-03.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2018-04.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2018-05.csv -P ./taxi_data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2018-06.csv -P ./taxi_data

#Weather data
#Open NYC weather data from https://www.ncdc.noaa.gov/
#TODO - Make a curl command to download the data

#Lat/long by address
#Open address data from ...
#TODO - Make a curl command to download the data (if needed??)


