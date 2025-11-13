'''
image_classifier_bing.py 

Version: This version addresses point 1 of the code review located at natsova/Programming/CodeReview/image_classifier_bing
and incorporates a Config class.

Automates dataset creation for image classification using Bing image search.  
Downloads, organizes, deduplicates, and validates images across predefined categories.  
Includes tools for visual review, selective deletion, and replacement of missing or corrupted images.  
Integrates fastai utilities for verification and Pillow for format conversions.  
Colab version: https://colab.research.google.com/drive/15SlbLCHlO7t5kd95vTeOOmmCIhzHKTH6

Author: Natalie Sova, 2025
'''

# ========================= Imports =========================

!pip install icrawler
!pip install bing-image-downloader --quiet
!pip install ipywidgets --quiet
!pip install pillow --quiet

from pathlib import Path
import shutil
from PIL import Image
import os
from bing_image_downloader import downloader
import time
from fastai.vision.all import *
import matplotlib.pyplot as plt
import random
from IPython.display import display
import ipywidgets as widgets
import io
import hashlib
import random
from dataclasses import dataclass
from typing import List

# ========================= Config class =========================

@dataclass
class Config:
    dataset_path: Path
    categories: List[str]
    images_per_search: int = 5
    images_per_category: int = 5
    sleep_time: int = 2
    remove_duplicates: bool = True

# Config instance
config = Config(
    dataset_path=Path("datasets"),
    categories=["sky", "beach", "umbrella", "dog", "book"],
    images_per_search = 5,
    images_per_category = 5,
    sleep_time = 2,
    remove_duplicates = True
)

# Function: Create folder for config.categories

def create_folders_for_categories(config: Config):
    if config.dataset_path.exists() and config.dataset_path.is_dir(): # Clear existing dataset
        shutil.rmtree(config.dataset_path)
        print("Dataset folder cleared!")

    config.dataset_path.mkdir(exist_ok=True)

    for category in config.categories: # Create category folders
        (config.dataset_path / category).mkdir(exist_ok=True)
        print("Created", config.dataset_path, "/", category)

# Function: Download images

def randomise_query(base):
    modifiers = [
        "high quality", "hdr", "aesthetic", "macro", "film", "close up",
        "dawn", "dusk", "natural light", "4k"
    ]
    return f"{base} {random.choice(modifiers)}"

def download_images(config: Config):

    for category in config.categories:
        category_path = config.dataset_path / category
        queries = [f"{category} photo", f"{category} sun photo", f"{category} night photo"]

        image_counter = 1

        for query in queries:
            if image_counter > config.images_per_category:
                break

            print(f"Downloading: {query}")
            temp_dir = category_path / "temp_download"

            try:
                for _ in range(3):
                    query = randomise_query(f"{category} photo")
                    print(f"Downloading: {query}")
                    downloader.download(
                        query,
                        limit=config.images_per_search,
                        output_dir=str(temp_dir),
                        adult_filter_off=True,
                        force_replace=False, # "True" has a bug - AttributeError: type object 'Path' has no attribute 'isdir'
                        timeout=60,
                        verbose=False
                    )
                    time.sleep(config.sleep_time)

                # Move images from temp folder to main category folder
                query_folder = temp_dir / query
                if query_folder.exists():
                    for img_file in query_folder.glob("*.*"):
                        if image_counter > config.images_per_category:
                            break
                        try:
                            with Image.open(img_file) as img:
                                img = img.convert("RGB")
                                img = img.resize((400,400), Image.LANCZOS)
                                save_path = category_path / f"{image_counter}.jpg"
                                img.save(save_path, "JPEG")
                                image_counter += 1
                        except Exception as e:
                            print(f"Skipped invalid: {img_file.name} ({e})")

                    shutil.rmtree(query_folder)

                # Remove temp folder if empty
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()

            except Exception as e:
                print(f"An unexpected error occurred during download for '{query}': {e}")
                continue

        # Remove duplicates
        remove_duplicate_images(config)

        print(f"{category} done: {image_counter - 1} images downloaded.")

def randomise_query(base):
    modifiers = [
        "high quality", "hdr", "aesthetic", "macro", "film", "close up",
        "dawn", "dusk", "natural light", "4k"
    ]
    return f"{base} {random.choice(modifiers)}"

def download_images_for_category(category, config, image_counter, needed=None):
    """Downloads and processes images for a single category."""
    category_path = config.dataset_path / category
    category_path.mkdir(parents=True, exist_ok=True)

    queries = [
        f"{category} photo",
        f"{category} sun photo",
        f"{category} night photo"
    ]

    temp_dir = category_path / "temp_download"
    images_added = 0
    needed = needed or config.images_per_category

    for query in queries:
        if image_counter > config.images_per_category or images_added >= needed:
            break

        print(f"Downloading: {query}")

        try:
            # Download multiple randomized versions of the same query
            for _ in range(3):
                randomized_query = randomise_query(f"{category} photo")
                print(f"Query: {randomized_query}")
                downloader.download(
                    randomized_query,
                    limit=config.images_per_search,
                    output_dir=str(temp_dir),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                    verbose=False
                )
                time.sleep(config.sleep_time)

                query_folder = temp_dir / randomized_query
                if not query_folder.exists():
                    continue

                for img_file in query_folder.glob("*.*"):
                    if image_counter > config.images_per_category or images_added >= needed:
                        break
                    try:
                        with Image.open(img_file) as img:
                            img = img.convert("RGB")
                            img = img.resize((400, 400), Image.LANCZOS)
                            save_path = category_path / f"{image_counter}.jpg"
                            img.save(save_path, "JPEG")
                            image_counter += 1
                            images_added += 1
                    except Exception as e:
                        print(f"Skipped invalid: {img_file.name} ({e})")

                shutil.rmtree(query_folder, ignore_errors=True)

            # Clean up empty temp folder
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

        except Exception as e:
            print(f"Error during download for '{query}': {e}")
            continue

    return image_counter, images_added

def download_images(config):
    """Initial dataset population."""
    for category in config.categories:
        print(f"\nProcessing category: {category}")
        image_counter = 1

        image_counter, _ = download_images_for_category(category, config, image_counter)

        remove_duplicate_images(config)
        print(f"{category} done: {image_counter - 1} images downloaded.")

def replace_deleted_images(config, recursion_level: int = 0, max_recursion_depth: int = 10):
    """Recursively refill missing images in categories."""
    if recursion_level > max_recursion_depth:
        print("Max recursion depth reached — stopping to avoid infinite loop.")
        return

    for category in config.categories:
        category_path = config.dataset_path / category
        category_path.mkdir(exist_ok=True)

        existing_images = list(category_path.glob("*.jpg"))
        count_existing = len(existing_images)
        needed = config.images_per_category - count_existing

        if needed <= 0:
            print(f"{category}: already has {count_existing} images.")
            continue

        print(f"\n{category}: {count_existing} found, need {needed} more.")
        image_counter = count_existing + 1

        image_counter, added = download_images_for_category(category, config, image_counter, needed)

        remove_duplicate_images(config)

        current_count = len(list(category_path.glob("*.jpg")))
        if current_count < config.images_per_category:
            print(f"{category}: Still under target ({current_count}/{config.images_per_category}). Retrying...")
            replace_deleted_images(config, recursion_level + 1)
        else:
            print(f"{category}: Now has {current_count} images.\n")

    if recursion_level == 0:
        print("Replacement process complete!")

# Function: Remove duplicates

def remove_duplicate_images(config: Config):
  if config.remove_duplicates:
      for category in config.categories:
          category_path = config.dataset_path / category
          hashes = {}
          for img_path in category_path.glob("*.jpg"):
              with open(img_path, "rb") as f:
                  file_hash = hashlib.md5(f.read()).hexdigest()
              if file_hash in hashes:
                  print(f"Removing duplicate: {img_path.name} (duplicate of {hashes[file_hash].name})")
                  img_path.unlink()
              else:
                  hashes[file_hash] = img_path

# Function: Display 5 random images per category

def display_images(config: Config):
  config.dataset_path = Path("datasets")

  for category in config.categories:
      category_path = config.dataset_path / category
      all_images = list(category_path.glob("*.jpg"))

      if not all_images:
          print(f"{category}: No images found!")
          continue

      sample_images = random.sample(all_images, min(5, len(all_images)))

      print(f"{category}:")
      plt.figure(figsize=(15, 3))
      for i, img_path in enumerate(sample_images, 1):
          img = Image.open(img_path)
          plt.subplot(1, 5, i)
          plt.imshow(img)
          plt.axis('off')
          plt.title(f"{img_path.name}")
      plt.show()

#  Function: Select invalid images for deletion

checkboxes = {}

def select_img_for_deletion(config: Config):
  config.dataset_path = Path("datasets")
  images_to_discard = []

  for category in config.categories:
      category_path = config.dataset_path / category
      all_images = sorted(category_path.glob("*.jpg"))
      if not all_images:
          continue

      print(f"\nCategory: {category}")
      container = widgets.VBox()  # vertical layout
      rows = []

      for img_path in all_images:
          img = Image.open(img_path).convert("RGB")
          img.thumbnail((150, 150))

          bio = io.BytesIO()
          img.save(bio, format="JPEG")
          bio.seek(0)

          img_widget = widgets.Image(value=bio.read(), format='jpeg', width=150, height=150)

          cb = widgets.Checkbox(value=True, description=img_path.name)
          checkboxes[img_path] = cb

          row = widgets.HBox([img_widget, cb])
          rows.append(row)

      container.children = rows
      display(container)

# Function: Delete unchecked images

def delete_unchecked_images():
    for img_path, cb in checkboxes.items():
        if not cb.value:  # unchecked = discard
            img_path.unlink()
            print(f"Discarded {img_path.name}")

    print("Cleanup finished!")

# Function: Remove corrupted images from dataset

def remove_corrupted_images(config: Config):
    failed = verify_images(get_image_files(config.dataset_path))
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} corrupted images.")

# # ========================= DatasetManager class ========================

class DatasetManager:
    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        create_folders_for_categories(self.config)
        download_images(self.config)
        remove_duplicate_images(self.config)
        select_img_for_deletion(self.config); replace_deleted_images(self.config)
        display_images(self.config)

# ================== Main entry point - Single workflow ==================

def main():
    manager = DatasetManager(config)
    manager.setup()

if __name__ == "__main__":
    main()


'''
For Colab (or Jupyter) it is better to split the code into cells for interactive 
# notebook control.

create_folders_for_categories(config)
download_images(config)
display_images(config)
select_img_for_deletion(config)
delete_unchecked_images()
replace_deleted_images(config)
display_images(config)
'''
