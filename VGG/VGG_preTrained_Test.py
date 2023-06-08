

import numpy as np
import cv2
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import img_to_array
from retinaface import RetinaFace

# Load the saved features and labels
features = np.load(r'AlexNet\features.npy')
labels = np.load(r'AlexNet\labels.npy')

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom classification layers on top of the pre-trained model
x = base_model.output
# Add your custom classification layers here

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Load your trained model weights if applicable

# Perform testing or prediction
test_image_path = r"C:\Users\Administrator\Downloads\20230520_132559.jpg"  # Set the path to your test image

# Load and preprocess the test image
image = cv2.imread(test_image_path)

# Perform face detection using RetinaFace
detector = RetinaFace(quality="best")
faces = detector.predict(image)

# Iterate through detected faces
for face in faces:
    # Extract the face region
    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
    face_org = image[y1:y2, x1:x2]

    
    # Preprocess the face image
    face = cv2.resize(face_org, (224, 224))  # Resize the face image to match the input size of the model
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Preprocess the face image
    face = img_to_array(face)
    face = preprocess_input(face)
    
    # Extract features
    face_features = base_model.predict(np.expand_dims(face, axis=0)).flatten()
    
    # Calculate distances between the test face features and all the loaded features
    distances = np.linalg.norm(features - face_features, axis=1)
    
    # Find the index of the best matching feature
    best_match_index = np.argmin(distances)
    
    # Get the corresponding label for the best matching feature
    label = labels[best_match_index]
    
    # Print the label
    print("Label: {}".format(label))
    
    # Display the face with label
    cv2.imshow(label, face_org)
    cv2.waitKey(0)
