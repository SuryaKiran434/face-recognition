import os
import pickle
from PIL import Image
import face_recognition
import pillow_heif

# Register HEIC support
pillow_heif.register_heif_opener()

def resize_image(image_path, max_width=800, max_height=800):
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))
        img.save(image_path)

def encode_surya_faces(dataset_path, output_file):
    known_encodings = []
    known_names = []

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)

        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            continue

        print(f"Processing {image_path}...")
        resize_image(image_path)  # Resize the image

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image, model="hog")  # Use "hog" for faster processing
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) > 0:
            known_encodings.append(face_encodings[0])
            known_names.append("Surya")

    with open(output_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print(f"Encodings saved to {output_file}")

if __name__ == "__main__":
    dataset_path = "preprocessed_surya"
    output_file = "face_encodings_Surya.pkl"
    encode_surya_faces(dataset_path, output_file)
