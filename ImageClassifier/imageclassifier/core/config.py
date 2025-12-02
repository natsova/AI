# core/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class Config:
    dataset_path=Path("datasets")
    categories: List[str]
    images_per_search: int = 50
    images_per_category: int = 150
    sleep_time: int = 2
    remove_duplicates: bool = True
