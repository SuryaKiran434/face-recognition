import os
import random
import shutil
import sys
from pathlib import Path

import click
import pillow_heif
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pillow_heif.register_heif_opener()


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")


def resize_image(image_path, max_width=800, max_height=800):
    """
    Resize the image to fit within the given dimensions and normalize to JPEG.
    Removes the original file if its extension differs from .jpg.
    Returns the path of the resulting file.
    """
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))
        rgb = img.convert("RGB")
        new_path = os.path.splitext(image_path)[0] + ".jpg"
        rgb.save(new_path, "JPEG", quality=90)
    if new_path != image_path:
        os.remove(image_path)
    return new_path


def sample_and_process(src_folder, dst_folder, max_samples):
    all_images = [
        img for img in os.listdir(src_folder)
        if img.lower().endswith(_IMAGE_EXTS)
    ]
    if not all_images:
        click.echo(f"No valid images found in {src_folder}. Skipping...")
        return

    sampled_images = random.sample(all_images, min(len(all_images), max_samples))

    for image_name in sampled_images:
        src_image_path = os.path.join(src_folder, image_name)
        dst_image_path = os.path.join(dst_folder, image_name)
        try:
            shutil.copy(src_image_path, dst_image_path)
            resize_image(dst_image_path)
            click.echo(f"Processed {image_name} to {dst_folder}")
        except Exception as e:
            click.echo(f"Error processing {src_image_path}: {e}", err=True)


def preprocess_datasets(src_dirs, dst_dir, max_samples_per_folder=50):
    os.makedirs(dst_dir, exist_ok=True)

    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            click.echo(f"Source directory {src_dir} does not exist. Skipping...")
            continue

        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)

            if os.path.isdir(item_path):
                dst_person_path = os.path.join(dst_dir, item)
                os.makedirs(dst_person_path, exist_ok=True)
                sample_and_process(item_path, dst_person_path, max_samples_per_folder)

            elif item.lower().endswith(_IMAGE_EXTS):
                dst_flat_path = os.path.join(dst_dir, os.path.basename(src_dir))
                os.makedirs(dst_flat_path, exist_ok=True)
                sample_and_process(src_dir, dst_flat_path, max_samples_per_folder)
                break  # flat-folder case handled once


@click.command()
@click.option("--src", "src_dirs", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Source dataset directory. Pass --src multiple times for multiple sources.")
@click.option("--dst", "dst_dir", required=True, type=click.Path(file_okay=False),
              help="Destination directory for preprocessed images.")
@click.option("--max-samples", default=50, show_default=True, type=int,
              help="Maximum number of images to sample per folder.")
def main(src_dirs, dst_dir, max_samples):
    preprocess_datasets(list(src_dirs), dst_dir, max_samples)


if __name__ == "__main__":
    main()
