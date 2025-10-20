'''
image_classifier_bing.py

Automates dataset creation for image classification using Bing image search.  
Downloads, organizes, deduplicates, and validates images across predefined categories.  
Includes tools for visual review, selective deletion, and replacement of missing or corrupted images.  
Integrates fastai utilities for verification and Pillow for format conversions.  
Original file is located at https://colab.research.google.com/drive/15SlbLCHlO7t5kd95vTeOOmmCIhzHKTH6

Author: Natalie Sova, 2025
'''

!pip install icrawler
!pip install bing-image-downloader --quiet
!pip install ipywidgets --quiet
!pip install pillow --quiet

# Import libraries
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

# Config
dataset_path = Path("dataset")
categories = ["bird", "forest", "serval", "dog", "book"]
images_per_search = 5
images_per_category = 5
sleep_time = 2
remove_duplicates = True

# Clear existing dataset
dataset_path = Path("dataset")
if dataset_path.exists() and dataset_path.is_dir():
    shutil.rmtree(dataset_path)
    print("Dataset folder cleared!")
else:
    print("No dataset folder found.")

# Create category folders
dataset_path.mkdir(exist_ok=True)
for category in categories:
    (dataset_path / category).mkdir(exist_ok=True)
    print(dataset_path, "/", category)

# Function: Download images
def randomise_query(base):
    modifiers = [
        "high quality", "hdr", "aesthetic", "macro", "film", "close up",
        "dawn", "dusk", "natural light", "4k"
    ]
    return f"{base} {random.choice(modifiers)}"

def download_images():
    for category in categories:
        category_path = dataset_path / category
        queries = [f"{category} photo", f"{category} sun photo", f"{category} night photo"]

        image_counter = 1

        for query in queries:
            if image_counter > images_per_category:
                break

            print(f"Downloading: {query}")
            temp_dir = category_path / "temp_download"

            try:
                for _ in range(3):
                    query = randomise_query(f"{category} photo")
                    print(f"Downloading: {query}")
                    downloader.download(
                        query,
                        limit=images_per_search,
                        output_dir=str(temp_dir),
                        adult_filter_off=True,
                        force_replace=False, # "True" has a bug - AttributeError: type object 'Path' has no attribute 'isdir'
                        timeout=60,
                        verbose=False
                    )
                    time.sleep(sleep_time)

                # Move images from temp folder to main category folder
                query_folder = temp_dir / query
                if query_folder.exists():
                    for img_file in query_folder.glob("*.*"):
                        if image_counter > images_per_category:
                            break
                        try:
                            with Image.open(img_file) as img:
                                img = img.convert("RGB")
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
        remove_duplicate_images()

        print(f"{category} done: {image_counter - 1} images downloaded.")

# Function: Remove duplicates
def remove_duplicate_images():
  if remove_duplicates:
      for category in categories:
          category_path = dataset_path / category
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
def display_images():
  dataset_path = Path("dataset")
  for category in categories:
      category_path = dataset_path / category
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

# Function: Preprocess images - convert to RGBA, resize, then convert to RGB
def convert_RGBA_RBB(dataset_path: Path):
    for img_path in dataset_path.rglob("*.*"):
        im = Image.open(img_path)
        if im.mode != "RGBA":
            im.convert("RGBA").save(img_path)

    resize_images(dataset_path, max_size=400, dest=dataset_path, recurse=True)

    for img_path in dataset_path.rglob("*.*"):
        im = Image.open(img_path)
        if im.mode != "RGB":
            im.convert("RGB").save(img_path)

# Function: Replace deleted or missing images (recursive)
max_recursion_depth = 10
target_images_per_category = 5
count_existing = 0
needed = target_images_per_category

def replace_deleted_images(recursion_level=0):
    if recursion_level > max_recursion_depth:
        print("Max recursion depth reached — stopping to avoid infinite loop.")
        return

    for category in categories:
        category_path = dataset_path / category
        category_path.mkdir(exist_ok=True)
        existing_images = list(category_path.glob("*.jpg"))
        count_existing = len(existing_images)
        needed = target_images_per_category - count_existing
        if needed <= 0:
            print(f"{category}: already has {count_existing} images.")
            continue

        print(f"\n{category}: {count_existing} found, need {needed} more.")
        image_counter = count_existing + 1

        queries = [
            f"{category} photo",
            f"{category} sun photo",
            f"{category} night photo"
        ]

        for query in queries:
            if needed <= 0:
                break
            print(f"Downloading replacements for {category}: '{query}'")
            temp_dir = category_path / "temp_download"
            try:
                downloader.download(
                    query,
                    limit=images_per_search,
                    output_dir=str(temp_dir),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=60,
                    verbose=False
            )
            except Exception as e:
                print(f"Error during download for '{query}': {e}")
                continue

            time.sleep(sleep_time)

            query_folder = temp_dir / query
            if query_folder.exists():
                for img_file in query_folder.glob("*.*"):
                    if needed <= 0:
                        break
                    try:
                        with Image.open(img_file) as img:
                            #img.verify()  # check if valid
                            #img = Image.open(img_file).convert("RGB")
                            save_path = category_path / f"{image_counter}.jpg"
                            img.save(save_path, "JPEG")
                            image_counter += 1
                            needed -= 1
                    except Exception as e:
                        print(f"Skipped invalid: {img_file.name} ({e})")

                shutil.rmtree(query_folder, ignore_errors=True)

            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

        # Cleanup duplicates after refill
        remove_duplicate_images()

        # Re-check total count
        current_count = len(list(category_path.glob("*.jpg")))
        if current_count < target_images_per_category:
            print(f"{category}: Still under target ({current_count}/{target_images_per_category}). Retrying...")
            replace_deleted_images(recursion_level + 1)
        else:
            print(f"{category}: Now has {current_count} images.\n")

    if recursion_level == 0:
        print("Replacement process complete!")

# Function: Select invalid images for deletion
checkboxes = {}

def select_img_for_deletion():
  dataset_path = Path("dataset")
  images_to_discard = []

  for category in categories:
      category_path = dataset_path / category
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

# List all folders and files in your dataset directory
!ls -R dataset
!ls -l

download_images()
display_images()
select_img_for_deletion()
delete_unchecked_images(); replace_deleted_images()

# Remove corrupted images from dataset
def remove_corrupted_images():
    failed = verify_images(get_image_files(dataset_path))
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} corrupted images.")
