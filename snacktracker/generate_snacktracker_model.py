
# ==================================================
#                       IMPORTS
# ==================================================

import argparse
import logging
import glob
import os.path as path
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import keras
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils  import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# ==================================================
#                      CONSTANTS
# ==================================================
CLASSES = [
    "PlantersPeanuts",
    "SmartFoodPopcorn"
]

#parse args
argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--logLevel", required=False, help="Set the logging level as DEGUG, INFO, WARNING, ERROR, or CRITICAL")
argParser.add_argument("-o", "--output", action='store_true', required=False, help="If present the log will be output to STDOUT as well as the default logfile")
argParser.add_argument("-t", "--trainFolder", required=True, help="This is the path to the folder that contains all the training images")
argParser.add_argument("-te", "--testFolder", required=True, help="This is the path to the folder that contains the testing images")


args = vars(argParser.parse_args())


#configure logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("snacktracker.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

if (args["output"] is not None):    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.INFO)

#The loging levels taken here are: DEGUG, INFO, WARNING, ERROR, CRITICAL
if (args["logLevel"] is not None):
    rootLogger.info("Setting loglevel to: " + args["logLevel"])
    rootLogger.setLevel(args["logLevel"])

#Get the training data
rootLogger.debug("Number of classes: %d", len(CLASSES))
rootLogger.debug("Classes are: %s", str(CLASSES))

#Grab the filepaths of all the training images
file_paths = glob.glob(path.join(args["trainFolder"], '*/*.jpeg'))

#Grab the filepaths of the testing images
testing_file_paths = glob.glob(path.join(args["testFolder"], '*/*.jpeg'))

#Actually load the images
images = [misc.imread(path) for path in file_paths]
images = np.asarray(images)
testimages = [misc.imread(path) for path in testing_file_paths]
testimages = np.asarray(testimages)

#Get the image sizes
nRows = images.shape[1]
nCols = images.shape[2]
nDims = images.shape[3]
rootLogger.info("Training image size is nRows: %d nCols: %d nDims: %d", nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

#Scale the images from their 0-255 color values to 0-1
images = images.astype('float32')
images = images / 255

testimages = testimages.astype('float32')
testimages = testimages / 255

#Now we need to actually get the labels from the file paths
n_images = images.shape[0]
labels = np.zeros(n_images)

rootLogger.debug("n_images length is: %d", n_images)
rootLogger.debug("labels length is: %d", len(labels))


for i in range(n_images):
    labels[i] = CLASSES.index(file_paths[i].split("/")[-2])

test_n_images = testimages.shape[0]
testlabels = np.zeros(test_n_images)

for i in range(test_n_images):
    testlabels[i] = CLASSES.index(testing_file_paths[i].split("/")[-2])

#Convert for Keras readable labels
labels_hot = to_categorical(labels)

test_labels_hot = to_categorical(testlabels)

rootLogger.debug("Original label 0 : %s", labels[0])
rootLogger.debug("After conversion to categorical : %s", labels_hot[0])

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CLASSES), activation='softmax'))


batch_size = 100
epochs = 1
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
#         zoom_range=0.2, # randomly zoom into images
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

model.summary()

history = model.fit_generator(datagen.flow(images, labels_hot, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(images.shape[0] / float(batch_size))),
                              epochs=epochs,
                              validation_data=(testimages, test_labels_hot),
                              workers=4)

#history = model.fit(images, labels_hot, batch_size = batch_size, epochs=epochs, verbose=1, validation_data=(testimages, test_labels_hot))

model.evaluate(testimages, test_labels_hot)

model.save("models/model_" + str(time.time()) + ".h5")

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


