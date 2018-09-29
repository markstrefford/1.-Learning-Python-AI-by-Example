"""
Face recognition pipeline

"""

import os
import cv2
import imutils
import cv2
import argparse
import sys
from glob import glob
from opencv_face_recognition import face_recognition

# TODO - Add a parameter to a folder of images
# TODO - Add a parameter to a pre-trained model
# TODO - Handle more than one face!

# Load the face detector
face_recogniser = face_recognition()

# Create a list of training images and labels
labels = []
images = glob('./my_photos/*/*', recursive=True)
for filename in images:
    labels.append(filename.split(os.path.sep)[-2].title())

# Train using this dataset
face_recogniser.train(images, labels)

# Now start video feed
cam = cv2.VideoCapture(0)
if ( not cam.isOpened() ):
    print ("no cam!")
    sys.exit()
print ("cam: ok.")

while True:
    ret, frame = cam.read()
    labelled_frame = face_recogniser.recognise(frame)

    cv2.imshow('Face Recognition', labelled_frame)
    k = cv2.waitKey(5) & 0xFF

    # bailout on 'esc'
    if k == 27:
        break














