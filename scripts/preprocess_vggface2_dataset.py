import os
import random
import shutil

def preprocess_vggface2(src_dir, dst_dir, num_samples=5):
    """
    Preprocess the VGGFace2 dataset by selecting a fixed number of images
    from each person's folder and copying them to the destination.

    Args:
        src_dir (str): Path to the source dataset folder (e.g., "known_faces/train").
        dst_dir (str): Path to the destination folder (e.g., "known_faces/processed").
        num_samples (int): Number of images to copy per person.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for person_folder in os.listdir(src_dir):
        person_path = os.path.join(src_dir, person_folder)
        if os.path.isdir(person_path):
            dst_person_path = os.path.join(dst_dir, person_folder)
            if not os.path.exists(dst_person_path):
                os.makedirs(dst_person_path)

            images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            sampled_images = random.sample(images, min(len(images), num_samples))

            for img in sampled_images:
                shutil.copy(os.path.join(person_path, img), os.path.join(dst_person_path, img))
                print(f"Copied {img} to {dst_person_path}")

if __name__ == "__main__":
    preprocess_vggface2(
        "/Users/suryakiran/Downloads/archive/train",  # Path to your VGGFace2 train folder
        "known_faces/processed_VGG",  # Destination for preprocessed data
        num_samples=5
    )
