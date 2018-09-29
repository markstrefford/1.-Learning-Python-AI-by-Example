"""
Face recognition pipeline

"""

import os
import cv2
import imutils
import cv2
import argparse

# Load the OpenCV face detector
protoPath = './opencv_face_recognition/face_detector/deploy.prototxt'
modelPath = './opencv_face_recognition/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the Face Recogniser
embedding_model = './opencv_face_recognition/openface/nn4.small2.v1.t7'
embedder = cv2.dnn.readNetFromTorch(embedding_model)



