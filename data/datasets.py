#!/usr/bin/env python3

import torch
from pathlib import Path
from torch.functional import Tensor
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
import itertools
from typing import Union, List
import pandas as pd
from dataprep import clean_datasets
from tqdm import tqdm
from torch.utils.data.dataset import Subset


wav_idx = 0
srate_idx = 1
trans_idx = 2
speaker_idx = 3
book_idx = 4
ut_idx = 5
sample_path_idx = 6


def write_trans(root: Path, sample_path: Path, id1: str, id2: str, id3: str,
                transcription: str, prefix: str = None) -> None:
    """Writes the transcription to the transcription file

    Args:
        root (Path): Folder in which the transcription csv will be written.
        sample_path (Path): Path to the audio file corresponding to provided transcription
        id1 (str): Speaker id.
        id2 (str): Book id.
        id3 (str): Utterance id.
        transcription (str): The transcription.
        prefix (str, optional): If provided, add prefix to saved csv file. By default file is saved as trans.csv.
    """
    transcription = transcription.lower()
    if prefix is None:
        trans_file = root / f"trans.csv"
    else:
        trans_file = root / f"{prefix}.trans.csv"

    if not trans_file.exists():
        trans_file.touch()
        trans_file.write_text(
            "path,speaker_id,book_id,utterance_id,transcription\n")

    with trans_file.open('a', encoding='utf-8') as f:
        f.write(",".join([str(sample_path), id1, id2,
                id3, f"{transcription}"]) + "\n")

    return


def write_trans_clean(dataset, dataset_str: str, target_trans: str):
    if isinstance(dataset, Subset):
        root = Path(dataset.dataset._path)
    else:
        root = Path(dataset._path)

    for sample in tqdm(dataset, f"writing transcription for {dataset_str}"):
        id1 = str(sample[speaker_idx])
        id2 = str(sample[book_idx])
        id3 = str(sample[ut_idx])
        trans = sample[trans_idx]
        sample_path = root / id1 / id2 / f"{id1}-{id2}-{id3:0>4}.flac"
        write_trans(Path(target_trans), sample_path,  id1,
                    id2, id3, trans, prefix=dataset_str)
def transcribe_libri_clean():

    for dataset_str, dataset in tqdm(clean_datasets.items(), desc="Writing transcriptions for clean datasets"):
        write_trans_clean(dataset, dataset_str, "/scratch/fharrathi/data" + "/LibriSpeech")



if __name__=="__main__":
    
    transcribe_libri_clean()

