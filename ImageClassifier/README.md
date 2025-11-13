GPT-5 code review for image_classifier_bing.py

# Review summary:

## Strengths
   * Good modular structure — most tasks (download, dedup, preprocessing, validation, etc.) are split into functions.
   * Visual interactivity via ipywidgets is a plus for Jupyter workflows.
   * Attention to data quality — duplicate removal, corruption checks, recursion for missing images.

## Areas for Improvement
   * Inconsistent reuse of configuration (duplicate global variable references, dataset_path resets, etc.).
   * Risk of breaking directories (temp folder handling, path concatenation in bing_image_downloader results).
   * Unreliable recursive downloading (can loop indefinitely if bing_image_downloader fails).
   * Missing exception handling in image I/O — potential resource leaks.
   * Some redundancy and unclear naming (e.g. convert_RGBA_RBB typo).
   * Performance: multiple unnecessary image conversions and resaves.
   * Logical bug potential—query_folder = temp_dir / query is not necessarily correct (Bing downloader subfolders may differ).

# Recommended Improvements:

## 1. Structure and Code Hygiene

Problem: dataset_path and categories are used globally and sometimes redefined. This makes testing harder.
Improvement:
  * Wrap everything inside a class or use a configuration dataclass.
  * Dependency-inject configuration instead of using globals.

### Solution:

```python
# ========================= Imports =========================

from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import List

# ========================= Config class =========================

@dataclass
class Config:
    dataset_path: Path
    categories: List[str]

# Config instance
config = Config(
    dataset_path=Path("datasets"),
    categories=["sky", "beach", "umbrella", "dog", "book"]
)

def create_folders_for_categories(config: Config):
    if config.dataset_path.exists() and config.dataset_path.is_dir(): # Clear existing dataset
        shutil.rmtree(config.dataset_path)
        print("Dataset folder cleared!")

    config.dataset_path.mkdir(exist_ok=True)

    for category in config.categories: # Create category folders
        (config.dataset_path / category).mkdir(exist_ok=True)
        print("Created", config.dataset_path, "/", category)


# ========================= DatasetManager class =========================

class DatasetManager:
    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        create_folders_for_categories(self.config)


# ========================= Main entry point =========================

def main():
    manager = DatasetManager(config)
    manager.setup()

if __name__ == "__main__":
    main()
```

## 2. Directory Handling in Downloads

Problem: downloader.download() creates its own folders inside output_dir. You’re hardcoding 
query_folder = temp_dir / query, which may break if the downloader changes its internal naming.
Improvement: 
  * Use dynamic folder detection.
  * Wrap temp_dir creation and cleanup in a context manager-style helper.

## 3. Image Validation and Duplication Handling

Your duplicate removal by MD5 hash is simple but can be expensive for large sets.
You can make it safer and more performant:
  * Hash only a small portion of the file or use perceptual hashing (e.g. imagehash).
  * Do not open all files unnecessarily — read binary data only

## 4. Preprocessing & RGB Conversion

There’s a small bug  in "def convert_RGBA_RBB(...):"
You meant convert_RGBA_RGB, plus you currently always convert to RGBA → resize → RGB, which is unnecessary.

## 5. Replace Deleted Images Logic

Issue: Recursive call to replace_deleted_images() can easily go deep if downloads fail due to network or 
API limits.
Improvement:
  * Replace recursion with a retry-count loop:

## 6. General Cleaning / Style

* Avoid Shadowing Built-ins: The variable name list is fine, but don’t reassign in your loops, e.g., don’t use
* list = [...].
* Naming: Fix typos like convert_RGBA_RBB.
* Logging: Replace print statements with Python’s logging for better control.
* Function Reuse: You’re redefining image download logic several times (main + replace). Refactor into a single
* reusable download_images_for(category) function.

## 7. Visualization Improvements

Your visualization is good for basic inspection. Consider adding the following that yields cleaner output when 
running for many categories.

  def preview_category_grid(category: str, n: int = 5):
      category_path = dataset_path / category
      imgs = random.sample(list(category_path.glob("*.jpg")), min(n, len(list(category_path.glob("*.jpg")))))
      fig, axes = plt.subplots(1, len(imgs), figsize=(15, 3))
      for ax, img_path in zip(axes, imgs):
          ax.imshow(Image.open(img_path))
          ax.axis("off")
      plt.suptitle(category)
    plt.show()

## 8. Performance Notes

  * Reduce repeated disk I/O in duplicate and corruption removal by batching.
  * Use ThreadPoolExecutor for hashed image reads if dataset is large and CPU-bound.
  * Store metadata (hash, category, filename) in a CSV once processed — reuse it in later cleaning steps.

## 9. Bonus: Folder Summary at the End

A quick summary printer is useful after downloads:
  
  def summarize_dataset(path: Path):
      print("\nDataset Summary:")
      for category in sorted(path.iterdir()):
          if category.is_dir():
              count = len(list(category.glob("*.jpg")))
              print(f"{category.name}: {count} images")
