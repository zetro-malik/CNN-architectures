import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# Set the path to the dataset folder
dataset_path = 'face_images'

# Define the input shape for the VGGNet model
input_shape = (224, 224, 3)

# Define the number of classes
num_classes = len(os.listdir(dataset_path))

# Set the path to save the trained weights
weights_path = 'vggnet_face_weights.h5'

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Create a new sequential model
model = Sequential()

# Add the pre-trained VGG16 model to the new sequential model
model.add(base_model)

# Add custom classification layers on top of the pre-trained model
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create an image data generator with data augmentation
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Generate augmented training data from the images in the dataset folder
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the trained weights
model.save_weights(weights_path)
