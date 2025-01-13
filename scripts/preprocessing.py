import os
import shutil
from PIL import Image
import pillow_heif

# Register HEIC support
pillow_heif.register_heif_opener()

def resize_image(image_path, max_width=800, max_height=800):
    """
    Resize the image to fit within the specified dimensions.
    """
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))
        img.save(image_path)

def preprocess_datasets(src_dirs, dst_dir):
    """
    Preprocess images from multiple datasets by copying them to the destination folder.
    Includes support for resizing and various image formats.

    Args:
        src_dirs (list): List of source dataset directories.
        dst_dir (str): Destination directory for preprocessed images.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for src_dir in src_dirs:
        for person_name in os.listdir(src_dir):
            person_path = os.path.join(src_dir, person_name)

            if os.path.isdir(person_path):  # Dataset with subdirectories for each person
                dst_person_path = os.path.join(dst_dir, person_name)
                if not os.path.exists(dst_person_path):
                    os.makedirs(dst_person_path)

                for image_name in os.listdir(person_path):
                    process_image(person_path, dst_person_path, image_name)
            else:  # Flat dataset with all images in one folder
                process_image(src_dir, dst_dir, person_name)

def process_image(src_dir, dst_dir, image_name):
    """
    Process a single image by copying and resizing.
    """
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
        src_image_path = os.path.join(src_dir, image_name)
        dst_image_path = os.path.join(dst_dir, image_name)
        shutil.copy(src_image_path, dst_image_path)
        resize_image(dst_image_path)
        print(f"Processed {image_name} to {dst_dir}")

if __name__ == "__main__":
    # List of source directories (can be updated based on your datasets)
    src_dirs = [
        "/Users/suryakiran/Downloads/archive/train",  # Example: VGGFace2 train dataset
        "/Users/suryakiran/Desktop/Surya"            # Example: Personal folder
    ]
    dst_dir = "known_faces"  # Unified destination folder

    preprocess_datasets(src_dirs, dst_dir)
