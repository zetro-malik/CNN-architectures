import numpy as np
import cv2
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import img_to_array
from retinaface import RetinaFace
import os



def onTestFolder():
    # Load the saved features and labels
    features = np.load(r'VGG\features.npy')
    labels = np.load(r'VGG\labels.npy')

    # Load the pre-trained VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False)

    # Add custom classification layers on top of the pre-trained model
    x = base_model.output
    # Add your custom classification layers here

    # Create the final model
    model = Model(inputs=base_model.input, outputs=x)

    # Load your trained model weights if applicable

    # Set the path to the test folder
    test_folder = r"Vegetable Images\test"

    # Initialize counters for total, correct, and wrong guesses for each class
    class_totals = {}
    class_correct_guesses = {}
    class_wrong_guesses = {}

    # Iterate through each subfolder (groundtruth label) in the test folder
    for label_folder in os.listdir(test_folder):
        label_path = os.path.join(test_folder, label_folder)
        if not os.path.isdir(label_path):
            continue
        
        # Initialize counters for the current class
        class_totals[label_folder] = 0
        class_correct_guesses[label_folder] = 0
        class_wrong_guesses[label_folder] = 0
        
        # Iterate through images in the subfolder
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            
            # Load and preprocess the test image
            image = cv2.imread(image_path)
            test_img = cv2.resize(image, (224, 224))  # Resize the image to match the input size of the model
            image = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Preprocess the image
            image = img_to_array(image)
            image = preprocess_input(image)
            
            # Extract features
            test_features = base_model.predict(np.expand_dims(image, axis=0)).flatten()
            
            # Calculate distances between the test features and all the loaded features
            distances = np.linalg.norm(features - test_features, axis=1)
            
            # Find the index of the best matching feature
            best_match_index = np.argmin(distances)
            
            # Get the corresponding label for the best matching feature
            predicted_label = labels[best_match_index]
            
            # Update counters for the current class
            class_totals[label_folder] += 1
            if predicted_label == label_folder:
                class_correct_guesses[label_folder] += 1
            else:
                class_wrong_guesses[label_folder] += 1
            
    

    # Print the results for each class
    for label_folder in class_totals:
        total_guesses = class_totals[label_folder]
        correct_guesses = class_correct_guesses[label_folder]
        wrong_guesses = class_wrong_guesses[label_folder]
        
        correct_percentage = (correct_guesses / total_guesses) * 100
        wrong_percentage = (wrong_guesses / total_guesses) * 100
        
        print("Class: ", label_folder)
        print("Total Guesses: ", total_guesses)
        print("Correct Guesses: ", correct_guesses)
        print("Wrong Guesses: ", wrong_guesses)
        print("Correct Percentage: {:.2f}%".format(correct_percentage))
        print("Wrong Percentage: {:.2f}%".format(wrong_percentage))
        print()
















def onTestFile():
    # Load the saved features and labels
    features = np.load(r'VGG\features.npy')
    labels = np.load(r'VGG\labels.npy')

    # Load the pre-trained VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False)

    # Add custom classification layers on top of the pre-trained model
    x = base_model.output
    # Add your custom classification layers here

    # Create the final model
    model = Model(inputs=base_model.input, outputs=x)

    # Load your trained model weights if applicable

    # Perform testing or prediction
    test_image_path = r"C:\Users\Administrator\Downloads\OIP.jpeg"  # Set the path to your test image

    # Load and preprocess the test image
    image = cv2.imread(test_image_path)



    # Preprocess the face image
    image = cv2.resize(image, (224, 224))  # Resize the face image to match the input size of the model
    test_img = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Preprocess the face image
    image = img_to_array(image)
    image = preprocess_input(image)

    # Extract features
    face_features = base_model.predict(np.expand_dims(image, axis=0)).flatten()

    # Calculate distances between the test face features and all the loaded features
    distances = np.linalg.norm(features - face_features, axis=1)

    # Find the index of the best matching feature
    best_match_index = np.argmin(distances)

    # Get the corresponding label for the best matching feature
    label = labels[best_match_index]

    # Print the label
    print("Label: {}".format(label))

    # Display the face with label
    cv2.imshow(label, test_img)
    cv2.waitKey(0)


if __name__ == "__main__":
     onTestFile()