import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array

# Set the path to your dataset
dataset_path = 'face_images'

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False)

# Initialize lists to store features and labels
features = []
labels = []

# Iterate through each class folder
class_folders = os.listdir(dataset_path)
for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    if not os.path.isdir(class_path):
        continue
    
    print(class_path)
    # Iterate through images in the class folder
    for filename in os.listdir(class_path):
        image_path = os.path.join(class_path, filename)
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        
        # Extract features
        features.append(base_model.predict(np.expand_dims(img, axis=0)).flatten())
        labels.append(class_folder)

# Convert the lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Save the features and labels as NumPy arrays
np.save('features.npy', features)
np.save('labels.npy', labels)
