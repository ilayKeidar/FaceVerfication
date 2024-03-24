import keras.utils
import tensorflow as tf
import os
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np

from siameseModel import sModel
from dataPipeline import DataPipeline

# setting GPU consumption to avoid oom
gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TrainModel:

    def __init__(self):

        # creates an instance of the siamese model
        self.siamese_model = sModel().make_siamese_model()

        # creates an instance of the Binary cross entropy loss function, with the default parameters
        self.binary_cross_loss = tf.losses.BinaryCrossentropy()

        # creates instance of the Adam optimizer with a learning rate of 0.0001
        learning_rate = 1e-4
        self.opt = tf.keras.optimizers.Adam(learning_rate)

        self.data_pipeline = DataPipeline()
        self.image_size = self.data_pipeline.image_size

        # creating the checkpoints -
        # to save the state of a model during training and allowing you to resume training from the same point

        # defines the directory where the checkpoints will be saved
        checkpoint_dir = './training_checkpoints3'

        # creates a fixed prefix for all the checkpoint files
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

        # creates a checkpoint object to save and restore the optimizer and the model
        self.checkpoint = tf.train.Checkpoint(opt=self.opt, siamese_model=self.siamese_model)

    # building train step function: making a prediction ->
    # calculating loss -> deriving gradients -> calculating new weights

    # compiles the function train_step code into a Tensorflow graph
    @tf.function
    def train_step(self, batch):

        with tf.GradientTape() as tape:

            # extracts the first two elements from batch (anchor and validation image)
            x1 = batch[0]
            x2 = batch[1]

            # extracts the third element from batch (label)
            y = batch[2]

            # obtains the predicted outputs (yhat) to the inputs (x)
            yhat = self.siamese_model([x1, x2], training=True)

            # calculates loss - compares the predicted output (yhat) to the label (y - 0/1)
            loss = self.binary_cross_loss(y, yhat)

        # calculates gradients
        grad = tape.gradient(loss, self.siamese_model.trainable_variables)

        # applies the updated weights and biases to the network
        self.opt.apply_gradients(zip(grad, self.siamese_model.trainable_variables))

        return loss

    def train(self, data, epochs):
        # loops through epochs
        for epoch in range(1, epochs + 1):
            print('\n Epoch {}/{}'.format(epoch, epochs))
            progbar = tf.keras.utils.Progbar(len(data))

            r = Recall()
            p = Precision()
            a = BinaryAccuracy()

            # loops through each batch
            loss = None
            for idx, batch in enumerate(data):
                loss = self.train_step(batch)
                yhat = self.siamese_model.predict([batch[0], batch[1]])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                a.update_state(batch[2], yhat)
                progbar.update(idx + 1)

            print("loss: ", loss.numpy(), " Recall: ", r.result().numpy(), " Precision: ", p.result().numpy(),
                  " Accuracy: ", a.result().numpy)

            # saving checkpoints
            if epoch % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def load_model(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(siamese_model=self.siamese_model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint is not None:
            checkpoint.restore(latest_checkpoint).expect_partial()

if __name__ == '__main__':
    # creates instance of the DataPipeline class
    data_pipeline = DataPipeline()

    # retrieves the train_data variable
    train_data = data_pipeline.training_data()
    model = TrainModel()
    model.siamese_model.save('siamese_model3.h5')

    # # TRAINING THE MODEL
    # EPOCHS = 10
    # model.train(train_data, EPOCHS)
