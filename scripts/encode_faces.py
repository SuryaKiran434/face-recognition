import os
import pickle
import face_recognition

def encode_faces(dataset_path, output_file):
    """
    Encodes all faces in the dataset directory and saves them to a file.

    Args:
        dataset_path (str): Path to the dataset (e.g., 'known_faces/processed').
        output_file (str): Path to save the encodings (e.g., 'face_encodings.pkl').
    """
    known_encodings = []
    known_names = []

    # Loop through each person's folder
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        # Loop through each image of the person
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            # Only process image files
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
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
    dataset_path = "known_faces/processed"  # Path to the processed dataset
    output_file = "face_encodings.pkl"      # Output file for the encodings
    encode_faces(dataset_path, output_file)
