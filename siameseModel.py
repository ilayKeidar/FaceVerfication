from keras.layers import Dense, Input
from keras.models import Model

from distanceLayer import L1Distance
from dataPipeline import DataPipeline
from ModelEngineering import ModelEngineering


class sModel:

    def __init__(self):
        # creates an instance of the embedding model
        self.embedding = ModelEngineering.make_embedding()
        self.data_pipeline = DataPipeline()
        self.image_size = self.data_pipeline.image_size

    def make_siamese_model(self):
        # anchor image in the neural network
        input_image = Input(name='input_img', shape=(self.image_size, self.image_size, 3))

        # validation (positive/negative) image in the neural network
        validation_image = Input(name='validation_img', shape=(self.image_size, self.image_size, 3))

        siamese_layer = L1Distance()
        siamese_layer._name = 'distance'

        # computes the distance between the embeddings using the siamese_layer as an instance of the L1Distance class
        distances = siamese_layer(self.embedding(input_image), self.embedding(validation_image))

        # takes the 4096 unit vector and turns it into a number output (1x1) (the sigmoid makes it a number between 0-1)
        classifier = Dense(1, activation='sigmoid')(distances)

        # returns model with input of the anchor and the positive/negative images,
        # and output of the classification layer
        return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

if __name__ == '__main__':
    siamese_model = sModel().make_siamese_model()
    print(siamese_model.summary())

