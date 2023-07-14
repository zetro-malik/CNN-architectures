import os
import tensorflow as tf
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras_preprocessing import image
import numpy as np


# TOOK 50 seconds in training in my laptop i7 8th gen, 16gb




# Load the pre-trained MobileNet model
model = MobileNet(weights='imagenet')

# Set the path to the main folder containing subfolders with label names
dataset_path = r'Vegetable Images\train'

# Initialize empty lists to store labels and embeddings
labels = []
embeddings = []

# Iterate through each subfolder (label/class)
for label_folder in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label_folder)
    if not os.path.isdir(label_path):
        continue
    
    print(label_path)
    # Iterate through images in the subfolder
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Get the predictions for the image
        preds = model.predict(x)
        embedding = preds.flatten()
        
        # Append the label and embedding to the lists
        labels.append(label_folder)
        embeddings.append(embedding)

# Convert the lists to NumPy arrays
labels = np.array(labels)
embeddings = np.array(embeddings)

# Save the labels and embeddings as NumPy arrays
np.save('labels.npy', labels)
np.save('embeddings.npy', embeddings)
