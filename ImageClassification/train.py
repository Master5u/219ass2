# import the necessary packages
from imagesearch.cnn.networks.lenet import LeNet
from imagesearch.cnn.networks.alexnet import AlexNet
from imagesearch.cnn.networks.hannet import HanNet
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from collections import Counter


#
# Here are three kind of cnn-networks
######################
network1 = LeNet
network2 = AlexNet
network3 = HanNet
#######################
#
# you can change networks here!
# the program will run current_network
# if you want to use HanNet, change to: Current_network = network3
#

#########################
#
Current_network = network1
#
##########################


# Step 1: Loading Data..............................................................
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
args = vars(ap.parse_args())

# get the MNIST dataset
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# print dataset information
print("Training data shape: ", trainData.shape)
print("Training labels shape: ", trainLabels.shape)
print("Test data shape", testData.shape)
print("Test labels shape", testLabels.shape)
print("Count number: ", Counter(trainLabels))

# show some data
plt.figure(facecolor='white')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(trainData[i], cmap="gray")
    plt.title("label %d" % trainLabels[i])
    plt.axis("off")
plt.show()

# Step 2: in another .py file

#.............................................................................


# Step 3: Train your CNN models................................................

# if we are using "channels first" ordering, then reshape the data
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))

# otherwise, we are using "channels last" ordering
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# transform the training and testing labels into vectors
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)

# you can change network on the top of the code
model = Current_network.build(numChannels=1, imgRows=28, imgCols=28,
                    numClasses=10,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# only train and evaluate the model if we are not loading a pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    history = model.fit(trainData, trainLabels, batch_size=128, epochs=20,
              verbose=1)

    # show a table about accuracy and loss
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['loss', 'acc'], loc='upper left')
    plt.show()

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)
#.............................................................................


# Step 4: Predict with Trained Model..............................................
# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image = (testData[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image = (testData[i] * 255).astype("uint8")

    # merge the channels into one image
    image = cv2.merge([image] * 3)

    # resize the image from a 28 x 28 image to a 96 x 96 image so we can better see it
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

    # show the image and prediction
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
                                                    np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)