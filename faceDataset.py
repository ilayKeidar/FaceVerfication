import cv2
import os
import uuid
import tensorflow as tf
import numpy as np


class FaceData:

    def __init__(self):
        # input of verification images (images of myself)
        self.Pos_Path = os.path.join('data', 'positive')

        # input of images of others (labelled faces in the wild library)
        self.Neg_Path = os.path.join('data', 'negative')

        # Reference input for comparing to the positive and negative
        self.Anc_Path = os.path.join('data', 'anchor')

        self.capture_size = 250

    def make_directories(self):
        os.makedirs(self.Pos_Path)
        os.makedirs(self.Neg_Path)
        os.makedirs(self.Anc_Path)

    def collect_images(self):
        # accessing the webcam
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            # flipping the image
            frame = cv2.flip(frame, 1)

            # cropping the webcam image to make it 250x250 pixels like the dataset images
            frame = frame[100:100 + self.capture_size, 300:300 + self.capture_size, :]
            cv2.imshow('Collecting Images', frame)

            # collecting anchors

            # if 'a' is pressed on the keyboard
            if cv2.waitKey(1) & 0XFF == ord('a'):

                # generates unique filenames for each image taken using uuid library in the Anchor directory
                img_name = os.path.join(self.Anc_Path, '{}.jpg'.format(uuid.uuid1()))

                # saves the frame with the generated filename
                cv2.imwrite(img_name, frame)

            # collecting positives

            # if 'p' is pressed on the keyboard
            if cv2.waitKey(1) & 0XFF == ord('p'):

                # generates unique filenames for each image taken using uuid library in the Positive directory
                img_name = os.path.join(self.Pos_Path, '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(img_name, frame)

            # closing the webcam

            # if 'q' is pressed on the keyboard
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # performs random small changes to the images to increase the dataset
    @staticmethod
    def data_augmentation(img):
        data = []

        for i in range(5):
            img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
            img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 2))
            img = tf.image.stateless_random_saturation(img, lower=0.8, upper=1, seed=(
                np.random.randint(100), np.random.randint(100)))
            img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(
                np.random.randint(100), np.random.randint(100)))

            data.append(img)

        return data

    # implements it in the directories
    def aug_implement(self, path):
        for file_name in os.listdir(os.path.join(path)):
            img_path = os.path.join(path, file_name)
            img = cv2.imread(img_path)
            augmented_images = self.data_augmentation(img)

            for image in augmented_images:
                cv2.imwrite(os.path.join(path, '{}.jpg'.format(uuid.uuid1())), image.numpy())

    def aug_implement_final(self):
        self.aug_implement(self.Anc_Path)
        self.aug_implement(self.Pos_Path)


if __name__ == '__main__':
    FaceData().collect_images()


