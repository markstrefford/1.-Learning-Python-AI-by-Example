"""
View the results of our trained model

"""

import cv2
import pandas as pd
import os
import argparse
from subprocess import call
from model.model import cnn

image_size = (78, 227, 3)

# Process command line arguments if supplied
parser = argparse.ArgumentParser(
        description='Run our model to view predicted steering angles')
show_default = ' (default %(default)s)'

parser.add_argument('-data-file', dest='data-file',
                    default='./data/data.txt',
                    help='File containing list of images and steering angles')

parser.add_argument('-model-file', dest='model-file',
                    default='./logs/weights-2018-11-02-11-00-18.hdf5',
                    help='File containing saved weights')

parser.add_argument('-data-dir', dest='data-dir',
                    default='./data/data',
                    help='Directory containing images')

args = vars(parser.parse_args())

model = cnn(input_shape=image_size)
model.load_weights(args['model-file'])

columns = ['image_name', 'angle', 'date', 'time']
df = pd.read_csv(args['data-file'], names=columns, delimiter=' ')

steering_wheel_img = cv2.imread('./data/steering_wheel_image.jpg',0)
rows,cols = steering_wheel_img.shape

smoothed_angle = 0
cv2.startWindowThread()

for i, sample in df.iterrows():
    image_path = os.path.join(args['data-dir'], sample['image_name'])
    image = cv2.imread(image_path)
    cropped = image[100:, :]
    resized = cv2.resize(cropped, (int(cropped.shape[1] / 2), int(cropped.shape[0] / 2)))
    angle = model.predict([resized])
    # call("clear")
    print('Predicted steering angle: {}'.format(angle))
    print('Actual steering angle: {}'.format(sample['angle']))
    print('Delta: {}'.format(angle - sample['angle']))
    cv2.imshow("frame", image)
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((angle - smoothed_angle)), 2.0 / 3.0) * (angle - smoothed_angle) / abs(angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(steering_wheel_img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)

cv2.destroyAllWindows()


