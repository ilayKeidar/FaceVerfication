from keras.layers import Dense, MaxPooling2D, Conv2D, Input, Flatten
from keras.models import Model

from dataPipeline import DataPipeline


class ModelEngineering:

    @staticmethod
    def make_embedding():

        data_pipeline = DataPipeline()

        # creates the input layer (the tensor of the image)
        inp = Input(shape=(data_pipeline.image_size, data_pipeline.image_size, 3))

        # first block

        # applies a convolutional layer with 64 filters of size (10,10), and a ReLU activation function to the input
        filter_amount1 = 64
        filter_size1 = 10
        c1 = Conv2D(filter_amount1, (filter_size1, filter_size1), activation='relu')(inp)

        # applies max pooling to c1, with a stride of 64, pool size of (2,2) and a same padding
        stride_size = 64
        pool_size = 2
        m1 = MaxPooling2D(stride_size, (pool_size, pool_size), padding='same')(c1)

        # second block

        filter_amount2 = 128
        filter_size2 = 7
        c2 = Conv2D(filter_amount2, (filter_size2, filter_size2), activation='relu')(m1)
        m2 = MaxPooling2D(stride_size, (pool_size, pool_size), padding='same')(c2)

        # third block

        filter_amount3 = 128
        filter_size3 = 4
        c3 = Conv2D(filter_amount3, (filter_size3, filter_size3), activation='relu')(m2)
        m3 = MaxPooling2D(stride_size, (pool_size, pool_size), padding='same')(c3)

        # fourth block

        filter_amount4 = 256
        filter_size4 = 4
        c4 = Conv2D(filter_amount4, (filter_size4, filter_size4), activation='relu')(m3)

        # flattens the output of c4 into a 1 dimension vector
        f1 = Flatten()(c4)

        # creates a dense layer with 4096 units and sigmoid activation function to the output of f1
        unit_amount = 4096
        d1 = Dense(unit_amount, activation='sigmoid')(f1)

        # returns a keras embedding model with the tensor of the image as the input and d1 as the output
        return Model(inputs=[inp], outputs=[d1], name='embedding')


if __name__ == '__main__':
    model = ModelEngineering().make_embedding()
    print(model.summary())
