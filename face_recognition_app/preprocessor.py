"""Preprocess raw face image datasets: sample, resize, normalize to JPEG."""

from __future__ import annotations

import logging
import os
import random
import shutil

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()


logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic")


def resize_image(image_path, max_width=800, max_height=800):
    """Resize image to fit within (max_width, max_height) and normalize to JPEG.

    Removes the original file when its extension differs from .jpg.
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
        logger.info("No valid images in %s; skipping", src_folder)
        return

    sampled = random.sample(all_images, min(len(all_images), max_samples))

    for image_name in sampled:
        src = os.path.join(src_folder, image_name)
        dst = os.path.join(dst_folder, image_name)
        try:
            shutil.copy(src, dst)
            resize_image(dst)
            logger.info("Processed %s -> %s", image_name, dst_folder)
        except Exception:
            logger.exception("Error processing %s", src)


def preprocess_datasets(src_dirs, dst_dir, max_samples_per_folder=50):
    os.makedirs(dst_dir, exist_ok=True)
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            logger.warning("Source directory %s does not exist; skipping", src_dir)
            continue

        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)
            if os.path.isdir(item_path):
                dst_person = os.path.join(dst_dir, item)
                os.makedirs(dst_person, exist_ok=True)
                sample_and_process(item_path, dst_person, max_samples_per_folder)
            elif item.lower().endswith(_IMAGE_EXTS):
                dst_flat = os.path.join(dst_dir, os.path.basename(src_dir))
                os.makedirs(dst_flat, exist_ok=True)
                sample_and_process(src_dir, dst_flat, max_samples_per_folder)
                break  # flat-folder case handled once
