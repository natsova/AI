# Kaggle version: https://www.kaggle.com/code/nataliexe/deep-learning-model-image-classifier

## Deep Learning Model - Image Classifier

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
    !pip install -Uqq fastai icrawler --use-deprecated=legacy-resolver

# Import libraries 
from fastai.vision.all import *
from PIL import Image
from pathlib import Path
from time import sleep
import shutil
from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin.google import GoogleFeeder, GoogleParser

# Define custom downloader and crawler for Google images 
class MyDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        filename = super().get_filename(task, default_ext).split(".")[0]
        return self.prefix + filename + ".png"

class MyCrawler(GoogleImageCrawler):
    def __init__(self, feeder_cls=GoogleFeeder, parser_cls=GoogleParser, downloader_cls=MyDownloader, prefix="", *args, **kwargs):
        super().__init__(feeder_cls, parser_cls, downloader_cls, *args, **kwargs)
        self.downloader.prefix = prefix

# Function to search and download images 
def search_images(term, max_images=30, folder_name="."):
    print(f"Searching for '{term}'")
    crawler = MyCrawler(prefix=term, storage={'root_dir': folder_name})
    crawler.crawl(keyword=term, max_num=max_images)

# Download dataset for multiple categories 
dataset_path = Path('dataset')
categories = ('forest', 'bird', 'serval', 'spoodle', 'book')
images_per_search = 20

for i in categories:
    dest = dataset_path / i
    dest.mkdir(exist_ok=True, parents=True)
    search_images(f"{i} photo", images_per_search, dest)
    sleep(5)
    search_images(f"{i} sun photo", images_per_search, dest)
    sleep(5)
    search_images(f"{i} night photo", images_per_search, dest)
    sleep(5)
    print(f"Downloaded images for {i}")

# Preprocess images: convert to RGBA, resize, then convert to RGB 
for img_path in dataset_path.rglob("*.png"):
    im = Image.open(img_path)
    if im.mode != "RGBA":
        im.convert("RGBA").save(img_path)

resize_images(dataset_path, max_size=400, dest=dataset_path, recurse=True)

for img_path in dataset_path.rglob("*.*"):
    im = Image.open(img_path)
    if im.mode != "RGB":
        im.convert("RGB").save(img_path)

# Remove corrupted images from dataset 
failed = verify_images(get_image_files(dataset_path))
failed.map(Path.unlink)
print(f"Removed {len(failed)} corrupted images.")

# Verify final dataset structure 
for folder in dataset_path.iterdir():
    print(folder.name, len(list(folder.glob('*'))), "images")

# Remove unwanted categories if present (optional)
unwanted_classes = ['cat', 'motorbike']
for cls in unwanted_classes:
    folder = dataset_path / cls
    if folder.exists():
        shutil.rmtree(folder)

# Create DataLoaders for training 
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(dataset_path, bs=32)

dls.show_batch(max_n=12)

# Build an image classification model (defined with pretrained weights)
learn = vision_learner(dls, resnet18, metrics=error_rate)    

# Train the model for 10 epochs
learn.fine_tune(10)

# Classify a new image using the trained model 
test_image = Image.open("/kaggle/working/dataset/spoodle/spoodle photo000001.png")
if test_image.mode != "RGBA":
    test_image = test_image.convert("RGBA")
test_image = test_image.convert("RGB")
test_image.to_thumb(256, 256)

# If you get a FileNotFoundError, progressively execute each of the commands until you find the file.
# import os
# print(os.listdir("/kaggle/working"))
# print(os.listdir("/kaggle/working/dataset"))
# print(os.listdir("/kaggle/working/dataset/spoodle"))

# Use the trained model to predict the class of the new image and display probabilities for all classes
classes = list(learn.dls.vocab)
predicted_class, _, probs = learn.predict(test_image)
print(f"This is a: {predicted_class}.")
for idx, ele in enumerate(classes):
    print(f"Probability it's a {ele}: {probs[idx]:.4f}")

# Export the trained model 
learn.export('model.pkl')
