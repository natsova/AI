# core/dataset_manager.py

from pathlib import Path
import os
import shutil
from PIL import Image
import hashlib
from fastai.vision.all import verify_images, get_image_files
from dataclasses import dataclass

class DatasetManager:
    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        create_folders_for_categories(self.config)
        download_images(self.config)
        select_img_for_deletion(self.config)
        refill_categories(self.config)
        display_images(self.config)
        create_dataloader(self.config)
