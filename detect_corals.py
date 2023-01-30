import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
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
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

# separate the images and labels from the training data
featureSet, labels = caer.sep_train(train, IMG_SIZE=(IMG_SIZE, IMG_SIZE))

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

# set AI training parameters
BATCH_SIZE = 32
EPOCHS = 10

# create a data generator to feed the training data into the model
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# create the AI model
# todo: play around with these layers to optimize model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Converts the 4D output of the Convolutional blocks to a 2D feature which can be read by the Dense layer
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))

# Output Layer
model.add(Dense(2, activation='sigmoid'))

print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# create a list of callbacks that will display the training progress of the model
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

# train the model
training = model.fit(train_gen,
                     steps_per_epoch=len(x_train) // BATCH_SIZE,
                     epochs=EPOCHS,
                     validation_data=(x_val, y_val),
                     validation_steps=len(y_val) // BATCH_SIZE,
                     callbacks=callbacks_list)

# save the model to a folder where it can be loaded from later
model.save("model")

"""## Testing"""

# supply paths to the images used to test the model
test_paths = ["branching.jpeg", "encrusting.jpg"]

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
    predictions = model.predict(caer.reshape(prepare(img), IMG_SIZE, 1))

    # print out the category with the highest probability
    print("Predicted", test_path, " is a ", corals[np.argmax(predictions[0])])

    # print the probabilities the model predicted for each label
    print(predictions[0])
