# core/dataset_manager.py

import os
import io
import time
import socket
import random
import shutil
import hashlib
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from dataclasses import dataclass

from fastai.vision.all import verify_images, get_image_files
from bing_image_downloader import downloader

import ipywidgets as widgets
from IPython.display import display, clear_output

from core.config import Config   # FIX: must import Config


class DatasetManager:
    def __init__(self, config: Config):
        self.config = config

    # ---------------------------------------------------------
    # Folder creation
    # ---------------------------------------------------------
    def create_folders(self):
        path = self.config.dataset_path

        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            print("Dataset folder cleared!")

        path.mkdir(exist_ok=True)

        for category in self.config.categories:
            (path / category).mkdir(exist_ok=True)
            print(f"Created {path}/{category}")

    # ---------------------------------------------------------
    # Main dataset download
    # ---------------------------------------------------------
    def download_all(self):
        for category in self.config.categories:
            print(f"\nProcessing category: {category}")
            image_counter = 1
            image_counter, _ = self._download_category(category, image_counter)
            print(f"{category} done: {image_counter - 1} images downloaded.")

    def _randomise_query(self, base: str) -> str:
        modifiers = [
            "high quality", "hdr", "aesthetic", "macro", "film", "close up",
            "dawn", "dusk", "natural light", "4k"
        ]
        return f"{base} {random.choice(modifiers)}"

    def _download_category(self, category, image_counter, needed=None):
        config = self.config
        category_path = config.dataset_path / category
        category_path.mkdir(parents=True, exist_ok=True)

        temp_dir = category_path / "temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)

        queries = [
            f"{category} photo",
            f"{category} sun photo",
            f"{category} night photo"
        ]

        images_added = 0
        needed = needed or config.images_per_category

        for query in queries:
            if image_counter > config.images_per_category or images_added >= needed:
                break

            print(f"Downloading: {query}")

            try:
                for _ in range(3):
                    randomised_query = self._randomise_query(f"{category} photo")
                    print(f"Query: {randomised_query}")

                    try:
                        downloader.download(
                            randomised_query,
                            limit=config.images_per_search,
                            output_dir=str(temp_dir),
                            adult_filter_off=True,
                            force_replace=False,
                            timeout=30,
                            verbose=False
                        )
                    except Exception as e:
                        print(f"Error downloading '{randomised_query}': {e}")
                        continue

                    time.sleep(config.sleep_time)

                    query_folder = temp_dir / randomised_query
                    if not query_folder.exists():
                        print(f"No folder found for '{randomised_query}', skipping.")
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

                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()

            except Exception as e:
                print(f"Error during download for '{query}': {e}")
                continue

        self.remove_corrupted()
        self.remove_duplicates()

        return image_counter, images_added

    # ---------------------------------------------------------
    # Refill incomplete categories
    # ---------------------------------------------------------
    def refill(self, max_rounds=5):
        for round_idx in range(max_rounds):
            print(f"\n--- Round {round_idx+1}/{max_rounds} ---")
            categories_filled = True

            for category in self.config.categories:
                category_path = self.config.dataset_path / category
                category_path.mkdir(exist_ok=True)

                existing = list(category_path.glob("*.jpg"))
                count_existing = len(existing)
                needed = self.config.images_per_category - count_existing

                if needed <= 0:
                    print(f"{category}: OK ({count_existing}/{self.config.images_per_category})")
                    continue

                categories_filled = False
                print(f"{category}: {count_existing} found, need {needed} more")

                image_counter = count_existing + 1
                self._download_category(category, image_counter, needed)

                self.remove_corrupted()
                self.remove_duplicates()

                new_count = len(list(category_path.glob("*.jpg")))
                print(f"{category}: now {new_count}/{self.config.images_per_category}")

            if categories_filled:
                print("\nAll categories filled.")
                return

        print("\nStopped after max rounds. Some categories may still be short.")

    # ---------------------------------------------------------
    # Duplicate & corruption handling
    # ---------------------------------------------------------
    def remove_duplicates(self):
        config = self.config
        if not config.remove_duplicates:
            return

        for category in config.categories:
            category_path = config.dataset_path / category
            hashes = {}
            for img_path in category_path.glob("*.jpg"):
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hashes:
                    print(f"Removing duplicate: {img_path.name}")
                    img_path.unlink()
                else:
                    hashes[file_hash] = img_path

    def remove_corrupted(self):
        failed = verify_images(get_image_files(self.config.dataset_path))
        failed.map(Path.unlink)
        print(f"Removed {len(failed)} corrupted images.")

    # ---------------------------------------------------------
    # Display sample images
    # ---------------------------------------------------------
    def display_samples(self):
        for category in self.config.categories:
            category_path = self.config.dataset_path / category
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
                plt.title(img_path.name)
            plt.show()
