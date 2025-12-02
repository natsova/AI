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

def create_folders_for_categories(config: Config):
    if config.dataset_path.exists() and config.dataset_path.is_dir(): # Clear existing dataset
        shutil.rmtree(config.dataset_path)
        print("Dataset folder cleared!")

    config.dataset_path.mkdir(exist_ok=True)

    for category in config.categories: # Create category folders
        (config.dataset_path / category).mkdir(exist_ok=True)
        print("Created", config.dataset_path, "/", category)

# Main dataset download controller.
def download_images(config):
    for category in config.categories:
        print(f"\nProcessing category: {category}")
        image_counter = 1
        image_counter, _ = download_images_for_category(category, config, image_counter)
        print(f"{category} done: {image_counter - 1} images downloaded.")

def randomise_query(base: str) -> str:
    modifiers = [
        "high quality", "hdr", "aesthetic", "macro", "film", "close up",
        "dawn", "dusk", "natural light", "4k"
    ]
    return f"{base} {random.choice(modifiers)}"

def download_images_for_category(category, config, image_counter, needed=None):
    # Handles downloading, validating, and saving images for one category.
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
                randomised_query = randomise_query(f"{category} photo")
                print(f"Query: {randomised_query}")

                try:
                    downloader.download(
                        randomised_query,
                        limit=config.images_per_search,
                        output_dir=str(temp_dir),
                        adult_filter_off=True,
                        force_replace=False,  # safer setting due to Path bug
                        timeout=30,
                        verbose=False
                    )
                except (TimeoutError, socket.timeout) as e:
                    print(f"Timeout while downloading '{randomised_query}': {e}")
                    continue
                except Exception as e:
                    print(f"Network error during '{randomised_query}': {e}")
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

    remove_corrupted_images(config)
    remove_duplicate_images(config)

    return image_counter, images_added

def refill_categories(config, max_rounds=5):
    for round_idx in range(max_rounds):
        print(f"\n--- Round {round_idx+1}/{max_rounds} ---")

        categories_filled = True

        for category in config.categories:
            category_path = config.dataset_path / category
            category_path.mkdir(exist_ok=True)

            existing = list(category_path.glob("*.jpg"))
            count_existing = len(existing)
            needed = config.images_per_category - count_existing

            if needed <= 0:
                print(f"{category}: OK ({count_existing}/{config.images_per_category})")
                continue

            categories_filled = False
            print(f"{category}: {count_existing} found, need {needed} more")

            image_counter = count_existing + 1
            image_counter, added = download_images_for_category(
                category,
                config,
                image_counter,
                needed,
            )

            remove_corrupted_images(config)
            remove_duplicate_images(config)

            new_count = len(list(category_path.glob("*.jpg")))
            print(f"{category}: now {new_count}/{config.images_per_category}")

        if categories_filled:
            print("\nAll categories filled.")
            return

    print("\nStopped after max rounds. Some categories may still be short.")

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

def display_images(config: Config):
  config.dataset_path = Path("datasets")

  for category in config.categories:
      category_path = config.dataset_path / category
      all_images = list(category_path.glob("*.jpg"))

      if not all_images:
          print(f"{category}: No images found!")
          continue

      sample_images = random.sample(all_images, min(5, len(all_images))) # Display 5 random images per category.

      print(f"{category}:")
      plt.figure(figsize=(15, 3))
      for i, img_path in enumerate(sample_images, 1):
          img = Image.open(img_path)
          plt.subplot(1, 5, i)
          plt.imshow(img)
          plt.axis('off')
          plt.title(f"{img_path.name}")
      plt.show()

checkboxes = {}    # global mapping: str(path) -> Checkbox widget

def select_img_for_deletion(config):
   # Populate global checkboxes mapping with widgets for manual review.
    global checkboxes
    checkboxes.clear()

    dataset_path = Path(config.dataset_path)
    for category in config.categories:
        category_path = dataset_path / category
        if not category_path.exists():
            print(f"No folder: {category_path}")
            continue

        all_images = sorted([p for p in category_path.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not all_images:
            print(f"No images in '{category}'")
            continue

        print(f"\nCategory: {category}")
        rows = []
        for img_path in all_images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((150, 150))
                    bio = io.BytesIO()
                    img.save(bio, format="JPEG")
                    bio.seek(0)
                    img_widget = widgets.Image(value=bio.read(), format='jpeg', width=150, height=150)
            except Exception as e:
                print(f"Could not load {img_path.name}: {e}")
                continue

            cb = widgets.Checkbox(value=True, description=img_path.name)
            checkboxes[str(img_path)] = cb
            row = widgets.HBox([img_widget, cb])
            rows.append(row)

        container = widgets.VBox(rows)
        display(container)

    print("\nUncheck images to delete, then run delete_unchecked_images().")

def delete_unchecked_images(clear_ui: bool = True):
    # Delete images that have been unchecked in the UI.

    global checkboxes
    if not checkboxes:
        print("No checkboxes found.")
        return

    deleted = 0
    failed = 0
    for path_str, cb in list(checkboxes.items()):
        try:
            if not cb.value:
                p = Path(path_str)
                p.unlink(missing_ok=True)
                deleted += 1
                print(f"Deleted: {p}")
        except Exception as e:
            failed += 1
            print(f"Failed to delete {path_str}: {e}")

    print(f"Done. Deleted: {deleted}. Failed: {failed}.")

    if clear_ui:
        # Remove widgets from output so user sees result and UI cleared
        clear_output(wait=True)
        print(f"Deleted: {deleted}. Failed: {failed}.")
        # Keep checkboxes empty to avoid accidental repeats
        checkboxes.clear()

def remove_corrupted_images(config: Config):
    failed = verify_images(get_image_files(config.dataset_path))
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} corrupted images.")
