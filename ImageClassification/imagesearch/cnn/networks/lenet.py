# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


# Step 2: Write two CNN models...............................


class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,
              activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)

        # First convolution, activation, pooling layers.
        model.add(Conv2D(20, 5, padding="same",
                        input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second convolution, activation, pooling layers.
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # First fully connection and activation layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # Second fully connection layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights

        model.summary()
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
