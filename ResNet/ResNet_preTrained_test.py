import os
import numpy as np
from keras.applications.resnet import ResNet50, preprocess_input
from keras_preprocessing import image
import cv2

def onTestFile():
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Set the path to the test image
    test_image_path = r"C:\Users\Administrator\Downloads\OIP.jpeg"

    # Load the saved labels and embeddings
    labels = np.load(r'ResNet\labels.npy')
    embeddings = np.load(r'ResNet\embeddings.npy')

    # Load and preprocess the test image
    img = image.load_img(test_image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the predictions for the test image
    preds = model.predict(x)
    test_embedding = preds.flatten()

    # Calculate distances between the test embedding and all the saved embeddings
    distances = np.linalg.norm(embeddings - test_embedding, axis=1)

    # Find the index of the closest matching embedding
    best_match_index = np.argmin(distances)

    # Get the corresponding label for the best matching embedding
    label = labels[best_match_index]

    # Load and preprocess the test image for face detection
    image_data = cv2.imread(test_image_path)

    # Iterate through detected faces
    # ...

    # Preprocess the face image for classification
    img = cv2.resize(image_data, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Get the predictions for the face image
    img_pred = model.predict(img)
    img_embeddings = img_pred.flatten()

    # Calculate distances between the face embedding and all the saved embeddings
    img_distances = np.linalg.norm(embeddings - img_embeddings, axis=1)

    # Find the index of the closest matching embedding for the face
    best_match_index = np.argmin(img_distances)

    # Get the corresponding label for the best matching embedding
    image_label = labels[best_match_index]

    # Display the label on the image
    cv2.putText(
        image_data,
        image_label,
        (100, 100 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Display the image with labels
    cv2.imshow("Image with Labels", cv2.resize(image_data, (500, 500), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def onTestFolder():
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Set the path to the test folder
    test_folder_path = r"Vegetable Images\test"

    # Load the saved labels and embeddings
    labels = np.load(r'ResNet\labels.npy')
    embeddings = np.load(r'ResNet\embeddings.npy')

    # Initialize variables for counting correct and wrong guesses
    total_guesses = 0
    correct_guesses = 0
    wrong_guesses = 0

    # Initialize a dictionary to store the counts for each class
    class_counts = {label: {'total': 0, 'correct': 0, 'wrong': 0} for label in set(labels)}

    # Iterate through subfolders (ground truth labels) in the test folder
    for subfolder in os.listdir(test_folder_path):
        subfolder_path = os.path.join(test_folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Iterate through images in the subfolder
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)

            # Load and preprocess the test image
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Get the predictions for the test image
            preds = model.predict(x)
            test_embedding = preds.flatten()

            # Calculate distances between the test embedding and all the saved embeddings
            distances = np.linalg.norm(embeddings - test_embedding, axis=1)

            # Find the index of the closest matching embedding
            best_match_index = np.argmin(distances)

            # Get the corresponding label for the best matching embedding
            predicted_label = labels[best_match_index]

            # Update the counts for the current class
            class_counts[subfolder]['total'] += 1
            if predicted_label == subfolder:
                class_counts[subfolder]['correct'] += 1
                correct_guesses += 1
            else:
                class_counts[subfolder]['wrong'] += 1
                wrong_guesses += 1

            # Update the total guesses count
            total_guesses += 1

    # Calculate the percentage of correct and wrong guesses for each class
    class_percentages = {label: {
        'correct_percentage': (class_counts[label]['correct'] / class_counts[label]['total']) * 100,
        'wrong_percentage': (class_counts[label]['wrong'] / class_counts[label]['total']) * 100
    } for label in labels}

    # Calculate the overall percentage of correct and wrong guesses
    correct_percentage = (correct_guesses / total_guesses) * 100
    wrong_percentage = (wrong_guesses / total_guesses) * 100

    # Print the results for each class
    for label in class_counts:
        print("Class: ", label)
        print("Total Guesses: ", class_counts[label]['total'])
        print("Correct Guesses: ", class_counts[label]['correct'])
        print("Wrong Guesses: ", class_counts[label]['wrong'])
        print("Correct Percentage: {:.2f}%".format(class_percentages[label]['correct_percentage']))
        print("Wrong Percentage: {:.2f}%".format(class_percentages[label]['wrong_percentage']))
        print()

    # Print the overall results
    print("Total Guesses: ", total_guesses)
    print("Correct Guesses: ", correct_guesses)
    print("Wrong Guesses: ", wrong_guesses)
    print("Correct Percentage: {:.2f}%".format(correct_percentage))
    print("Wrong Percentage: {:.2f}%".format(wrong_percentage))


if __name__ == "__main__":
    onTestFile()
