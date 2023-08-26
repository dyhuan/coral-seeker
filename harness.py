import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

# set the image size in pixels
IMG_SIZE = 200

# number of rgb color channels
channels = 3

# folder path to training data
training_data_folder_path = "processed_images"

# get the training data subfolders
training_data_folders = os.listdir(training_data_folder_path)

# get the coral subfolders in the training data folder
corals = os.listdir(training_data_folder_path)

# get the training data from the coral folders
train = caer.preprocess_from_dir(
    training_data_folder_path,
    corals,
    channels=channels,
    IMG_SIZE=(IMG_SIZE, IMG_SIZE),
    isShuffle=True
)

# show an example image from the training data
# plt.figure(figsize=(30, 30))
# plt.imshow(train[0][0], cmap='gray')
# plt.show()

# separate the images and labels from the training data
featureSet, labels = caer.sep_train(train, IMG_SIZE=(IMG_SIZE, IMG_SIZE), channels=channels)

# normalize the images
featureSet = caer.normalize(featureSet)

# generate categories from the coral labels
labels = to_categorical(labels, len(corals))

# split the training data into training and validation partitions
x_train, x_val, y_train, y_val = train_test_split(featureSet, labels)

# delete unused variables to save memory
del train
del featureSet
del labels
gc.collect()

def train_model(model, batch_size, epochs):
    # create a data generator to feed the training data into the model
    datagen = canaro.generators.imageDataGenerator()
    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

    
    # create a list of callbacks that will display the training progress of the model
    callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

    # train the model
    training = model.fit(train_gen,
                     steps_per_epoch=len(x_train) // batch_size,
                     epochs=epochs,
                     validation_data=(x_val, y_val),
                     validation_steps=len(y_val) // batch_size,
                     callbacks=callbacks_list)

    # save the model to a folder where it can be loaded from later
    # model.save("model")

# supply paths to the images used to test the model
test_paths = ["branching.jpeg", "encrusting.jpg"]

def test_model_arbitrary(model):
    # loop through each test image
    for test_path in test_paths:

       # load the image into python
        img = cv.imread(test_path)

        # prepare the image to be analyzed with the ai
        def prepare(img):
            ratio = 200 / min(img.shape[0], img.shape[1])
            new_shape = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
            img = cv.resize(img, new_shape)
            img = img[img.shape[0] // 2 - 100:img.shape[0] // 2 + 100, img.shape[1] // 2 - 100:img.shape[1] // 2 + 100]
            return img

        # show the image
        plt.imshow(prepare(img))
        plt.show()

        # generate predictions for what the image is
        predictions = model.predict(caer.reshape(prepare(img), (IMG_SIZE, IMG_SIZE), 3))

        # print out the category with the highest probability
        print("Predicted", test_path, " is a ", corals[np.argmax(predictions[0])])

        # print the probabilities the model predicted for each label
        print(predictions[0])

def test_model(model): 
    # create a data generator to feed the validation data into the model
    datagen = canaro.generators.imageDataGenerator()
    test_gen = datagen.flow(x_val, y_val)

    # test the model
    results = model.evaluate(test_gen)
    return results