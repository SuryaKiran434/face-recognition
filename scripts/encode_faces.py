import os
import pickle
import face_recognition

def encode_faces(dataset_path, output_file):
    """
    Encodes all faces in the dataset directory and saves them to a file.

    Args:
        dataset_path (str): Path to the dataset (e.g., '/Users/suryakiran/Preprocessed_Faces').
        output_file (str): Path to save the encodings (e.g., '/Users/suryakiran/FaceRecognitionData/face_encodings.pkl').
    """
    known_encodings = []
    known_names = []

    # Get list of person folders and sort them in alphabetical order
    person_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

    # Loop through each person's folder
    for person_name in person_folders:
        person_folder = os.path.join(dataset_path, person_name)

        # Loop through each image of the person
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            # Only process image files (skip directories and non-image files)
            if not os.path.isfile(image_path) or not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            print(f"Processing {image_path}...")

            # Load the image
            image = face_recognition.load_image_file(image_path)
            # Detect face locations and encode faces
            face_locations = face_recognition.face_locations(image, model="cnn")  # Use "cnn" for better accuracy
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) > 0:
                known_encodings.append(face_encodings[0])
                known_names.append(person_name)

    # Save encodings to a file
    with open(output_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print(f"Encodings saved to {output_file}")

if __name__ == "__main__":
    # External paths outside the project directory
    dataset_path = "/Users/suryakiran/Preprocessed_Faces"  # Path to the processed dataset
    output_file = "/Users/suryakiran/FaceRecognitionData/face_encodings.pkl"  # Path to save encodings
    
    encode_faces(dataset_path, output_file)
