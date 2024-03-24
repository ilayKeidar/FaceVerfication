from keras.layers import Layer
import tensorflow as tf


# combines the two streams of inputs - anchor and validation images - into one (siamese neural network)
# inheriting from the 'Layer' class from Keras
class L1Distance(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # calculates the difference between the two streams of data - anchor and the positive/negative embeddings
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
