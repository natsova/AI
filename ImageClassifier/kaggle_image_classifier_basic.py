## Deep Learning Model - Image Classifier (Kaggle version, basic)

# Check for internet connectivity 
import socket
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error:
    raise Exception("No internet. Enable your internet connection.")

# Install dependencies if running on Kaggle 
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    !pip install fastai icrawler --upgrade --no-deps

# Import libraries 
!pip install icrawler
!pip install bing-image-downloader --quiet
!pip install ipywidgets --quiet
!pip install pillow --quiet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import shutil
from PIL import Image
import os
from bing_image_downloader import downloader
from icrawler.builtin import BingImageCrawler
import time
from fastai.vision.all import *
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import io
import hashlib
import random
from dataclasses import dataclass
from typing import List
import socket
import uuid
from pathlib import Path
from PIL import Image
import shutil
import time
from icrawler.builtin import BingImageCrawler
from fastai.vision.all import PILImage

def create_folders_for_categories():
    if dataset_path.exists() and dataset_path.is_dir(): # Clear existing dataset
        shutil.rmtree(dataset_path)
        print("Dataset folder cleared!")

    dataset_path.mkdir(exist_ok=True)

    for c in category: # Create category folders
        (dataset_path / c).mkdir(exist_ok=True)
        print("Created", dataset_path, "/", c)

create_folders_for_categories()

dataset_path = Path("datasets")
categories = ["sky", "ocean", "umbrella", "dog", "book"]
images_per_category = 15  # total images per category
images_per_query = 10     # images to try per query
resize_to = (400, 400)

def download_images_for_categories():
    for c in categories:
        category_path = dataset_path / c
        category_path.mkdir(parents=True, exist_ok=True)

        temp_dir = category_path / "temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)

        queries = [f"{c} photo", f"{c} sun photo", f"{c} night photo"]

        image_counter = 0

        for query in queries:
            if image_counter >= images_per_category:
                break

            print(f"Downloading: {query}")
            try:
                crawler = BingImageCrawler(storage={"root_dir": str(temp_dir)})
                crawler.crawl(keyword=query, max_num=images_per_query)
            except Exception as e:
                print(f"Network error during search '{query}': {e}")
                continue

            # Process images
            for img_file in temp_dir.glob("*.*"):
                if image_counter >= images_per_category:
                    break
                try:
                    with Image.open(img_file) as img:
                        img = img.convert("RGB")
                        img = img.resize(resize_to, Image.LANCZOS)
                        save_path = category_path / f"{image_counter}.jpg"
                        img.save(save_path, "JPEG")
                        image_counter += 1
                except Exception as e:
                    print(f"Skipped invalid: {img_file.name} ({e})")

            # Clear temp folder for next query
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

            time.sleep(2)  # slight pause to avoid being blocked

        print(f"Downloaded {image_counter} images for category '{c}'")

download_images_for_categories()

# Remove corrupted images from dataset 
failed = verify_images(get_image_files(dataset_path))
failed.map(Path.unlink)
print(f"Removed {len(failed)} corrupted images.")

# Verify final dataset structure 
for folder in dataset_path.iterdir():
    print(folder.name, len(list(folder.glob('*'))), "images")

# Create DataLoaders for training 
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(dataset_path, bs=32)

# View some images
dls.show_batch(max_n=12)

# Build an image classification model (defined with pretrained weights)
learn = vision_learner(dls, resnet18, metrics=error_rate)    

# Train the model for 10 epochs
learn.fine_tune(10)

# Confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# # Apply trained model
# Process the image through the same training pipeline
url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcontent.lyka.com.au%2Ff%2F1016262%2F4288x2848%2F6fef92978a%2Fspoodle-puppy.jpeg%2Fm%2F1280x0%2Ffilters%3Aformat(webp)&f=1&nofb=1&ipt=aff6f2e8af6a16d4ca4e06da3fb681d12344283830ffd571b25039e60182e01c"
test_image = PILImage.create(BytesIO(requests.get(url).content))
pred_class, pred_idx, outputs = learn.predict(test_image)
print(pred_class, outputs)

# Use the trained model to predict the class of the new image and display probabilities for all classes
classes = list(learn.dls.vocab)
predicted_class, _, probs = learn.predict(test_image)
print(f"This is a: {predicted_class}.")
for idx, ele in enumerate(classes):
    print(f"Probability it's a {ele}: {probs[idx]:.4f}")

# Export the trained model 
learn.export('model.pkl')
