import torchaudio
import torch
import sys
sys.path.append('src')
import click
from pathlib import Path
import random
from typing import List, Union, Dict
from tqdm import tqdm


def split_dataset(dataset, percentage: float):
    assert percentage > 0 and percentage < 1, "Unvalid percentage provided"
    total_count = len(dataset)
    train_count = int(percentage * total_count)
    split_index = _get_split_index(dataset, train_count)

    train_set = torch.utils.data.Subset(dataset, list(range(split_index)))
    val_set = torch.utils.data.Subset(
        dataset, list(range(split_index, len(dataset))))
    return train_set, val_set

def _get_split_index(dataset, start_index):
    split_index = start_index
    speaker_at_split = dataset[start_index][3]
    speaker = speaker_at_split

    while speaker == speaker_at_split:
        speaker = dataset[split_index][3]
        split_index += 1
    return split_index

train_tmp = torchaudio.datasets.LIBRISPEECH(root='/scratch/fharrathi/data', url="train-clean-100", download=True)

clean_datasets = {"train-clean-100": train_tmp,
                  "dev-clean": torchaudio.datasets.LIBRISPEECH(root='/scratch/fharrathi/data', url="dev-clean", download=True),
                  "test-clean": torchaudio.datasets.LIBRISPEECH(root='/scratch/fharrathi/data', url="test-clean", download=True),
                  }

