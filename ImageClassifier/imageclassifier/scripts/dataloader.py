# core/dataloader.py

from fastai.vision.all import DataBlock, ImageBlock, CategoryBlock, RandomSplitter, get_image_files, Resize
from core.config import Config
from pathlib import Path

def create_dataloader(config: Config, batch_size: int = 4, img_size: int = 192):
    """
    Creates a FastAI DataLoaders object for the given dataset path and categories.
    
    Args:
        config (Config): Configuration object containing dataset_path and categories.
        batch_size (int): Batch size for the DataLoader.
        img_size (int): Target size for image resizing.
    
    Returns:
        DataLoaders: FastAI DataLoaders object for training/validation.
    """
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        raise FileNotFoundError(f"No data found in {dataset_path}. Make sure the dataset is prepared.")

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=lambda x: x.parent.name,
        item_tfms=[Resize(img_size)]
    )

    dls = dblock.dataloaders(dataset_path, bs=batch_size)
    
    # Optional: print dataset stats
    print(f"Training set size: {len(dls.train_ds)}")
    print(f"Validation set size: {len(dls.valid_ds)}")
    print(f"Categories: {dls.vocab}")

    return dls
