"""

"""


import os
import imutils
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import sys
sys.path.append('../')
from utils import load_image_as_array

confidence = 0.5
mean_subtract_values = (104, 177, 123)


class face_embedding:
    def __init__(self, prototxt_path='../opencv_face_recognition/face_detector/deploy.prototxt',
                 model_path='../opencv_face_recognition/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel',
                 embedding_model='../opencv_face_recognition/openface/nn4.small2.v1.t7',
                 labels_file=False, recogniser_file=False):
        """

        :param prototxt_path: string Path to deploy.protext
        :param model_path: string Path to trained caffe model
        :param embedding_model: string Path to OpenFace model
        :param labels: Labels from trained classifier
        :param recogniser: Trained classifer
        """

        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.embedder = cv2.dnn.readNetFromTorch(embedding_model)
        self.labels = []
        self.embeddings = {'names': [], 'embeddings': []}
        if labels_file:
            self.le = pickle.load(open(labels_file, 'rb'))
        else:
            self.le = LabelEncoder()
        if recogniser_file:
            self.recogniser = pickle.load(open(recogniser_file, 'rb'))
        else:
            self.recogniser = SVC(C=1.0, kernel="linear", probability=True)

    def get_face_embeddings_from_image(self, image):
        """
        Return the bbox and the embeddings vector for the face in the image
        (Note we return this for the face that has the highest confidence from the detector)
        :param image: np.array
        :return: tuple, array
        """
        vec = []
        start_x, start_y, end_x, end_y = 0, 0, 0, 0

        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            mean_subtract_values, swapRB=False, crop=False)
        (h, w) = image.shape[:2]

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        self.detector.setInput(image_blob)
        predictions = self.detector.forward()
        # ensure at least one face was found
        if len(predictions) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(predictions[0, 0, :, 2])
            conf = predictions[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if conf > confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[start_y:end_y, start_x:end_x]

                # ensure the face width and height are sufficiently large
                if face.shape[0] >= 20 and face.shape[1] > 20:
                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(face_blob)
                    vec = self.embedder.forward()
        return (start_x, start_y, end_x, end_y), vec

    def _build_embeddings(self, X, y):
        for i, (image_path, name) in enumerate(zip(X, y)):
            image = load_image_as_array(image_path)
            print('Processing image {}'.format(image_path))
            _, face_embedding_vector = self.get_face_embeddings_from_image(image)
            if face_embedding_vector != []:
                self.embeddings['names'].append(name)
                self.embeddings['embeddings'].append(face_embedding_vector[0])

    def train(self, X, y):
        """
        Train the model based on the data provided
        :param X: list of filenames
        :param y: labels
        :return:
        """
        self._build_embeddings(X, y)
        self.labels = self.le.fit_transform(self.embeddings['names'])
        self.recogniser.fit(self.embeddings['embeddings'], self.labels)

    def predict(self, image):
        """
        Predict all faces in an image
        :param image: np.array containing image
        :return: np.array containing image, bbox and labels
        """
        box, vec = self.get_face_embeddings_from_image(image)
        predictions = self.recogniser.predict_proba(vec)[0]
        j = np.argmax(predictions)
        proba = predictions[j]
        name = self.le.classes_[j]

        start_x, start_y, end_x, end_y = box
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (start_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return image






