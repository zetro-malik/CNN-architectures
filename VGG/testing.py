import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import  img_to_array
from retinaface import RetinaFace

# Define the input shape for the VGGNet model
input_shape = (224, 224, 3)
label_names = ['KHUBAIB', 'OZAIFA', 'UZAIR', 'ZEESHAN']
# Set the path to the saved weights
weights_path = r'VGG\vggnet_face_weights.h5'

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
model.add(Dense(4, activation='softmax'))

# Load the trained weights
model.load_weights(weights_path)

# Load and preprocess the test image
test_image_path = r"C:\Users\Administrator\Downloads\WhatsApp Image 2023-05-20 at 8.28.34 PM.jpeg"  # Replace with the path to your test image
image = cv2.imread(test_image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection using RetinaFace
detector = RetinaFace(quality="best")
faces = detector.predict(rgb_image)

# Iterate through detected faces
for face in faces:
    # Extract the face region
    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
    face_image = rgb_image[y1:y2, x1:x2]

    # Preprocess the face image
    face_image = cv2.resize(face_image, input_shape[:2])
    face_image = img_to_array(face_image)
    face_image = preprocess_input(face_image)
    face_image = np.expand_dims(face_image, axis=0)

    # Perform prediction on the face image
    prediction = model.predict(face_image)

    # Get the predicted label
    predicted_label = np.argmax(prediction)

    # Print the predicted label
    print("Predicted Label:", label_names[predicted_label])
