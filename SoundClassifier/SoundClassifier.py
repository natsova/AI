# Deep learning model for classifying cat and dog sounds.
# - Converts .wav files to mel spectrogram images
# - Retrieves labels from a CSV in the dataset
# - Trains a classifier on the spectrogram images
#
# Kaggle notebook: https://www.kaggle.com/code/nataliexe/deep-learning-model-sound-classifier
# Dataset: https://www.kaggle.com/datasets/kitonbass/vgg-sound-only-cat-and-dog-sounds
# Code inspired by fast.ai lessons 1–2

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

# Import dependencies
import os
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from fastai.vision.all import *
import torchaudio
from torchaudio.transforms import MelSpectrogram
import numpy as np
import pandas as pd

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
dataset_root = Path("/kaggle/input/vgg-sound-only-cat-and-dog-sounds")
save_root    = Path("/kaggle/working/spectrograms")
save_root.mkdir(exist_ok=True, parents=True)

# # Process dataset

# Convert audio to mel-spectrogram
def audio_to_melspectrogram(file_path, save_dir, img_size=(128,128)):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=img_size[0])
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())
    
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{file_path.stem}.png"
    
    plt.figure(figsize=(2,2))
    plt.axis('off')
    librosa.display.specshow(S_norm, sr=sr, cmap='viridis')
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    #print(f"Saving spectrogram to: {out_file}")
    plt.close()

# # Re-label wav files and create mel-spectrograms

# Training dataset
train_csv = pd.read_csv(dataset_root / "train.csv")

# Function to process a CSV subset
def process_audio_csv(csv_df, audio_folder, split_name):
    #print(f"\nProcessing {split_name}...")
    countCsv = 0
    countMel = 0
    countMis = 0
    for idx, row in csv_df.iterrows():
        file_id = str(row['Unnamed: 0'])
        label   = row['Label'].replace(" ", "_")
        audio_file = dataset_root / audio_folder / f"{file_id}.wav"
        save_dir = save_root / split_name / label
        countCsv += 1
        
        if audio_file.exists():
            audio_to_melspectrogram(audio_file, save_dir)
            countMel += 1
        else:
            countMis += 1
            pass
            #print(f"Missing file: {audio_file}")
    print(
    f"{'Items in CSV:':25} {countCsv:>5}\n"
    f"{'Missing audio files:':25} {countMis:>5}\n"
    f"{'Spectrograms processed:':25} {countMel:>5}")
    print("[ CSV - missing =", countCsv-countMis,"]")

# Process train set
process_audio_csv(train_csv, "audio_train", "train")

# Create DataLoader
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(192, method='squish')
)

dls = dblock.dataloaders(save_root, bs=32)
dls.show_batch(max_n=6, nrows=2, ncols=3)

# Create model
learn = vision_learner(dls, resnet18, metrics=error_rate)

# Train model
learn.fine_tune(5)

# Confusion matrix to visualise accuracy of classifications
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Save and export model
learn.export('model_sound.pkl')
print("Model saved as 'model_sound.pkl'")
