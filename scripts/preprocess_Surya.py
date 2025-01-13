import os
import shutil

def preprocess_surya_folder(src_dir, dst_dir):
    """
    Preprocess images from the 'Surya' folder by copying them to the destination folder.
    Includes support for .heic files without conversion.

    Args:
        src_dir (str): Path to the folder containing personal photos (e.g., "Surya").
        dst_dir (str): Path to the destination folder.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Iterate through all files in the source directory
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)

        # Check if the file is an image with the supported formats
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            shutil.copy(file_path, os.path.join(dst_dir, file_name))
            print(f"Copied {file_name} to {dst_dir}")

if __name__ == "__main__":
    src_dir = "/Users/suryakiran/Desktop/Surya"  # Path to Surya folder
    dst_dir = "known_faces/preprocessed_surya"  # Destination for preprocessed data

    preprocess_surya_folder(src_dir, dst_dir)
