"""
Interface with AWS Rekognition

Upload images to train
Upload images to test / recognise faces

"""

import boto3
import urllib
import json

from config import bucket_name

def train(images):
    """
    Train AWS Rekognition on our training images
    images = [('path/image', 'label'), ('path/image', 'label')...]
    :param images:
    :return:
    """
    s3 = boto3.resource('s3')

    for image in images:
        file = open(image, 'rb')
        object = s3('')