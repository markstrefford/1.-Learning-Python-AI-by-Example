"""
Interface with AWS Rekognition

Upload images to train
Upload images to test / recognise faces

"""

import os
import io
import boto3
import urllib
import json
from PIL import Image


def train(image_paths, labels, bucket_name):
    """
    Train AWS Rekognition on our training images
    images = [('path/image', 'label'), ('path/image', 'label')...]
    :param images: array of tuples
    :param bucket_name: string
    :return:
    """
    s3 = boto3.resource('s3')

    for image, name in zip(image_paths, labels):
        print(image, name)
        file = open(image, 'rb')
        _, image_name = os.path.split(image)
        obj = s3.Object(bucket_name, 'images/' + image_name)
        ret = obj.put(Body=file,
                      Metadata={'FullName': name}
                      )
        print(ret)

def test(image):
    """
    Test our face recognition capabilities. Pass in the path to an image and the bucket_name
    :param image: string
    :param bucket_name: string
    :return:
    """
    print('testing image {}'.format(image))
    rekognition = boto3.client('rekognition', region_name='eu-west-1')
    dynamodb = boto3.client('dynamodb', region_name='eu-west-1')

    image = Image.open(image)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    image_binary = stream.getvalue()

    print('Called rekognition...')
    response = rekognition.search_faces_by_image(
        CollectionId='applied-ai-collection',
        Image={'Bytes': image_binary}
    )
    print(response)
    for match in response['FaceMatches']:
        print(match['Face']['FaceId'], match['Face']['Confidence'])

        face = dynamodb.get_item(
            TableName='applied_ai_collection',
            Key={'RekognitionId': {'S': match['Face']['FaceId']}}
        )

        if 'Item' in face:
            print(face['Item']['FullName']['S'])
        else:
            print('no match found in person lookup')

    return response['FaceMatches']




