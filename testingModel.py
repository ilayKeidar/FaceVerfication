import os
import matplotlib.pyplot as plt
import cv2
import struct
import numpy as np
import ast
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope

from dataPipeline import DataPipeline
from trainingModel import TrainModel
from faceDataset import FaceData
from ModelEngineering import ModelEngineering
from distanceLayer import L1Distance



class TestModel:

    def __init__(self, tModel):
        self.model = tModel
        self.make_embeddings = ModelEngineering.make_embedding()
        self.test_data = DataPipeline().testing_data()
        self.test_input, self.test_validation, self.y_true = self.test_data.as_numpy_iterator().next()
        self.yhat = None

        self.det_threshold = 0.7
        self.ver_threshold = 0.8

    def make_predictions(self):
        self.yhat = self.model.siamese_model.predict([self.test_input, self.test_validation])

        # post-processing the results to make the values close to 1, 1 and the values close to 0, 0
        result = []
        for prediction in self.yhat:
            if prediction > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result

    @staticmethod
    def verify(model, detection_threshold, verification_threshold):
        results = []
        for image in os.listdir(os.path.join('app_data', 'verification')):
            input_img = DataPipeline().preprocess(os.path.join('app_data', 'input', 'input_img.jpg'))
            validation_img = DataPipeline().preprocess(os.path.join('app_data', 'verification', image))

            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # calculates the detection amount, the amount of predictions greater than detection threshold
        detection = np.sum(np.array(results) > detection_threshold)

        # calculates the verification score as the ratio of detections to the total number of verification images
        verification = detection / len(os.listdir(os.path.join('app_data', 'verification')))

        # verified only if the verification amount is greater than the verification threshold
        verified = verification > verification_threshold

        return results, verified, detection, verification

    def webcam_testing(self):
        # accessing the webcam
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            # flipping the image
            frame = cv2.flip(frame, 1)

            # cropping the webcam image to make it 250x250 pixels like the dataset images
            frame = frame[100:100 + FaceData().capture_size, 300:300 + FaceData().capture_size, :]
            cv2.imshow('FaceID', frame)

            # press v to take a verification image
            if cv2.waitKey(1) & 0XFF == ord('v'):
                cv2.imwrite(os.path.join('app_data', 'input', 'input_img.jpg'), frame)
                results, verified, detection, verification = self.verify(self.model, self.det_threshold, self.ver_threshold)
                print("Verification Result: ", verified)
                print(detection)
                print(results)
                break

            # press q to quit
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return 0


if __name__ == '__main__':
    train_model = TrainModel()
    checkpoint_dir = './training_checkpoints4'
    with custom_object_scope({'L1Distance': L1Distance}):
        siamese_model = load_model('trained_model3.h5')

    t_model = TestModel(siamese_model)
    t_model.webcam_testing()

    #TestModel(t_model).save_embeddings()

    #pred = t_model.make_predictions()

    # # checking accuracy
    # correct = np.sum(np.array(pred) == np.array(t_model.y_true))
    # accuracy = correct / len(pred) * 100
    #print(accuracy)

    # print(pred)
    # print(t_model.y_true)
    #
    # # size of the images shown
    # plt.Figure(figsize=(10, 8))
    #
    # # specifying the place it is shown
    # plt.subplot(1, 2, 1)
    #
    # i = 0
    # plt.imshow(t_model.test_input[i])
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(t_model.test_validation[i])
    # plt.show()

