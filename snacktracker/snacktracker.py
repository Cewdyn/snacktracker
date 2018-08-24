
# ==================================================
#                       IMPORTS
# ==================================================

import argparse
import logging
import cv2
from keras.models import load_model
from keras.preprocessing import image



# ==================================================
#                      CONSTANTS
# ==================================================
CLASSES = [
    "PlantersPeanuts",
    "SmartFoodPopcorn",
    "NatureValleyAlmond",
    "Pistachios",
    "PopCorners",
    "SunChips",
    "SweetPotatoChips",
    "NutrigrainBlueberry",
    "RXBARPeanutButter",
    "ZooAnimalCrackers",
    "SnackWellsCremeSandwich",
    "KarsAlmonds",
    "StacysCinnamonPita",
    "CheeseIts"
]
IMG_SIZE = { "HEIGHT": 84, "WIDTH": 150}

#parse args
argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--logLevel", required=False, help="Set the logging level as DEGUG, INFO, WARNING, ERROR, or CRITICAL")
argParser.add_argument("-o", "--output", action='store_true', required=False, help="If present the log will be output to STDOUT as well as the default logfile")
argParser.add_argument("-m", "--model", required=True, help="Path to the Keras model that is to be used for prediction")
argParser.add_argument("-i", "--image", required=False, help="Path to the image you want to classify")

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

#Load the model we've created
model = load_model(args["model"])

rootLogger.debug("Getting image to classify")
if (args["image"] is not None):
    rootLogger.info("Image argument present. Opening file at %s", args["image"])
    img = cv2.imread(args["image"])
else:
    rootLogger.info("Image argument absent. Attempting to capture image from camera.")
    capture = cv2.VideoCapture(0)
    if (not capture.isOpened()):
        rootLogger.error("Image argument was not specified and there was no camera detected. Exiting.")
        exit()
    ret, img = capture.read()
    capture.release()

rootLogger.info("Image obtained. Displaying for confirmation.")
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.resize(img, dsize=(IMG_SIZE["WIDTH"], IMG_SIZE["HEIGHT"]))
img = img.astype('float32')
img = img / 255

rootLogger.info("Beggining prediction")
prediction = model.predict_classes(img[None,:,:,:])
probabilities = model.predict_proba(img[None,:,:,:])

rootLogger.info("Predicted image as class: %s", str(prediction))
rootLogger.info("Predicted image as: %s", CLASSES[prediction[0]])
probabilities = probabilities[0]
rootLogger.info("Probabilites is: %s", str(probabilities))
for i in range(probabilities.size):
    prob = probabilities[i]
    rootLogger.info("Prediction for %s is %.2f", CLASSES[i], prob)