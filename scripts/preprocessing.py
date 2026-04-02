import os
import shutil
import random
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

def preprocess_datasets(src_dirs, dst_dir, max_samples_per_folder=50):
    """
    Preprocess a sample of images from multiple datasets by copying them to the destination folder.
    Includes support for resizing and various image formats.

    Args:
        src_dirs (list): List of source dataset directories.
        dst_dir (str): Destination directory for preprocessed images.
        max_samples_per_folder (int): Maximum number of images to sample per folder.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} does not exist. Skipping...")
            continue

        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)

            if os.path.isdir(item_path):  # Subdirectory (person folder)
                dst_person_path = os.path.join(dst_dir, item)
                if not os.path.exists(dst_person_path):
                    os.makedirs(dst_person_path)

                sample_and_process(item_path, dst_person_path, max_samples_per_folder)

            elif item.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):  # Flat folder (images directly)
                dst_flat_path = os.path.join(dst_dir, os.path.basename(src_dir))
                if not os.path.exists(dst_flat_path):
                    os.makedirs(dst_flat_path)
                sample_and_process(src_dir, dst_flat_path, max_samples_per_folder)
                break  # Prevent redundant processing for flat datasets

def sample_and_process(src_folder, dst_folder, max_samples):
    """
    Randomly sample and process a limited number of images from a folder.

    Args:
        src_folder (str): Source folder path.
        dst_folder (str): Destination folder path.
        max_samples (int): Maximum number of images to sample.
    """
    all_images = [img for img in os.listdir(src_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
    if not all_images:
        print(f"No valid images found in {src_folder}. Skipping...")
        return

    sampled_images = random.sample(all_images, min(len(all_images), max_samples))

    for image_name in sampled_images:
        src_image_path = os.path.join(src_folder, image_name)
        dst_image_path = os.path.join(dst_folder, image_name)

        try:
            shutil.copy(src_image_path, dst_image_path)
            resize_image(dst_image_path)
            print(f"Processed {image_name} to {dst_folder}")
        except Exception as e:
            print(f"Error processing {src_image_path}: {e}")

if __name__ == "__main__":
    # List of source directories (update as needed)
    src_dirs = [
        "/Users/suryakiran/Downloads/archive/train",  # Example: VGGFace2 train dataset
        "/Users/suryakiran/FaceRecognitionData/Surya" # Example: Personal folder
    ]
    dst_dir = "/Users/suryakiran/Preprocessed_Faces"  # Unified destination folder
    max_samples_per_folder = 50  # Limit to 50 sampled images per folder

    preprocess_datasets(src_dirs, dst_dir, max_samples_per_folder)
