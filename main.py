import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

training_datafiles   =  pd.read_csv("/home/landon/Development/nvidia-deep-learning/data/sign_mnist_train.csv")
validation_datafiles =  pd.read_csv("/home/landon/Development/nvidia-deep-learning/data/sign_mnist_valid.csv")

print(training_datafiles.head())

# Extracting the data

label_training = training_datafiles['label']
label_validation = validation_datafiles['label']
del training_datafiles['label']
del validation_datafiles['label']

# Extracting the images

image_training = training_datafiles.values
image_validation = validation_datafiles.values

''' Summarizing Data
print(image_training.shape)
print(image_validation.shape)
print(label_training.shape)
print(label_validation.shape)
'''

# Normalize the model

image_training = image_training / 255
image_validation = image_validation / 255

# Build the model
num_classes = 24

label_training = tf.keras.utils.to_categorical(label_training, num_classes)
label_validation = tf.keras.utils.to_categorical(label_validation, num_classes)

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# Summarize the model
model.summary

#Compile the model
 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model

model.fit(image_training, label_training, epochs=20,verbose=1, validation_data=(image_validation, label_validation))
