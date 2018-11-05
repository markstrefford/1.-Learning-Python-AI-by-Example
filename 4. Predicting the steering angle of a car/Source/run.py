"""
View the results of our trained model

"""

import cv2
import pandas as pd
import numpy as np
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

# model = cnn(input_shape=image_size)
# model.load_weights(args['model-file'])

columns = ['image_name', 'angle', 'date', 'time']
df = pd.read_csv(args['data-file'], names=columns, delimiter=' ')

steering_wheel_img = cv2.imread('./data/steering_wheel_image.jpg')
steering_wheel_w, steering_wheel_h, _ = steering_wheel_img.shape

smoothed_angle = 0
cv2.startWindowThread()

for i, sample in df.iterrows():
    image_path = os.path.join(args['data-dir'], sample['image_name'])
    image = cv2.imread(image_path)
    # Set up image to predict steering angle
    cropped = image[100:, :]
    resized = cv2.resize(cropped, (int(cropped.shape[1] / 2), int(cropped.shape[0] / 2)))
    # Determine predicted angle and delta from actual angle  
    angle = 0 # model.predict([resized])
    actual = sample['angle']
    delta = angle - actual
    # Make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    # and the predicted angle
    smoothed_angle += 0.2 * pow(abs((angle - smoothed_angle)), 2.0 / 3.0) * (angle - smoothed_angle) / abs(angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((steering_wheel_h/2,steering_wheel_w/2),-smoothed_angle,1)
    dst = cv2.warpAffine(steering_wheel_img,M,(steering_wheel_h,steering_wheel_w))
    # Create single image to display - road on the left, rotated steering wheel on the right
    display_img = np.zeros((image.shape[1], image.shape[0] + steering_wheel_w, 3))
    display_img[0:, 0:image.shape[0], :] = image
    display_img[0:, image.shape[1]:, :] = dst
    # call("clear")
    cv2.putText(image, 'Predicted angle = {}'.format(angle), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, 'Actual angle = {}'.format(actual), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, 'Delta (predicted - actual) = {}'.format(delta), (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    print('Predicted steering angle: {}'.format(angle))
    print('Actual steering angle: {}'.format(sample['angle']))
    print('Delta: {}'.format(angle - sample['angle']))
    cv2.imshow("Steering Demo", display_img)

cv2.destroyAllWindows()


