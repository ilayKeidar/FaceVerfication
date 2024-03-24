import tensorflow as tf

from faceDataset import FaceData


class DataPipeline:

    def __init__(self):
        # takes a specific amount of images from each dataset - anchor, positive, and negative
        pics_amount = 1250

        self.faceData = FaceData()
        self.anchor = tf.data.Dataset.list_files(self.faceData.Anc_Path + '\\*.jpg').take(pics_amount)
        self.positive = tf.data.Dataset.list_files(self.faceData.Pos_Path + '\\*.jpg').take(pics_amount)
        self.negative = tf.data.Dataset.list_files(self.faceData.Neg_Path + '\\*.jpg').take(pics_amount)

        self.image_size = 100

        # creates a labelled dataset where each element is [anchor image, positive image, 1]
        self.positives = tf.data.Dataset.zip(
            (self.anchor, self.positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(self.anchor)))))

        # creates a labelled dataset where each element is [anchor image, negative image, 0]
        self.negatives = tf.data.Dataset.zip(
            (self.anchor, self.negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(self.anchor)))))

        # concatenates the positive and negative datasets into a single dataset
        self.data = self.positives.concatenate(self.negatives)

        # variables for training_data function
        self.train_data = None
        self.training_percent = 0.7
        self.batch_size = 32
        self.prefetch_amount = 8

        self.test_data = None

    def preprocess(self, file_path):

        # reads the file path a returns a tensor of the bytes
        byte_img = tf.io.read_file(file_path)

        # decodes the JPEG image from bytes into a tensor
        img = tf.io.decode_jpeg(byte_img)

        # resizing to be 100x100px
        img = tf.image.resize(img, (self.image_size, self.image_size))

        # normalizing the pixel values of the image (values from 0 to 1)
        img = img / 255.0
        return img

    # the input is the data which is (input_img, validation_img, label)
    def preprocess_twin(self, input_img, validation_img, label):
        # input_img - anchor, validation_img - positive/negative, label-1/0
        return self.preprocess(input_img), self.preprocess(validation_img), label

    def data_preprocess(self):

        # applies preprocessing to each element in the dataset
        self.data = self.data.map(self.preprocess_twin)

        # caching the dataset for faster reading and training
        self.data = self.data.cache()

        # shuffles the elements of the dataset randomly -
        # ensures the learning algorithm doesn't accidentally learn any unintended relationships between samples
        self.data = self.data.shuffle(buffer_size=4000)

        return self.data

    # training partition - the part of the data used for training
    def training_data(self):

        # takes a certain amount of the dataset to use for the training, to save the rest for validation and testing
        self.train_data = self.data_preprocess().take(round(len(self.data) * self.training_percent))

        # creates batches of the data
        self.train_data = self.train_data.batch(self.batch_size)

        # prefetches a certain amount of data - allows overlapping of the model training and the data loading
        # for the next batch, as well as minimizing bottlenecks
        self.train_data = self.train_data.prefetch(self.prefetch_amount)

        return self.train_data

    def testing_data(self):

        # testing partition - the part of the data used for testing

        # skips the part of the dataset used for the training
        self.test_data = self.data_preprocess().skip(round(len(self.data) * self.training_percent))

        # takes the part of the dataset that is left
        self.test_data = self.test_data.take(round(len(self.data) * (1 - self.training_percent)))

        self.test_data = self.test_data.batch(self.batch_size)

        self.test_data = self.test_data.prefetch(self.prefetch_amount)
        return self.test_data


if __name__ == '__main__':
    data = DataPipeline().data
    samples = data.as_numpy_iterator()
    example = samples.next()
    print(DataPipeline().preprocess_twin(*example))



