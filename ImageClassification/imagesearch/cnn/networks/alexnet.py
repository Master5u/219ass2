# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

# Step 2: Write two CNN models...............................


class AlexNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,
              activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)

        # The first layer of convolutional network, using 96 convolution kernels, the size is 11x11 steps to 4,
        # the required picture is 227x227, 3 channels, no edges, the activation function uses relu
        model.add(Conv2D(96, (11, 11), strides=(1, 1), input_shape=inputShape, padding='same', activation='relu',
                         kernel_initializer='uniform'))
        # max pooling layer
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # The second layer is edged with 256 5x5 convolution kernels, and the activation function is relu
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # max pooling layer
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Third convolution layer, 384 of convolution kernels, size 3x3
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # Fourth convolution layer
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        # Fifth convolution layer
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()

        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
