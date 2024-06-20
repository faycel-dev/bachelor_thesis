import librosa
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, WhisperProcessor,AutoModelForSpeechSeq2Seq, AutoProcessor #,get_linear_schedule_with_warmup
from tqdm import tqdm
from SequenceToSequence import DataCollatorSpeechSeq2SeqWithPadding
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor
from pytorch_lightning.tuner  import Tuner
from model.WhisperLight import WhisperModel,base_model
import jiwer as jw
import re
#import whisper
#from whisper.normalizers import EnglishTextNormalizer
from functools import partial
from config import Config
import numpy as np
import eer
import os

device = 'cuda' if torch.cuda.is_available  else 'cpu'


def apdateID(speaker_id):
   list_speakres={19,   26,   27,   32,   39,   40,   60,   78,   83,   87,   89,
        103,  118,  125,  150,  163,  196,  198,  200,  201,  211,  226,
        229,  233,  248,  250,  254,  289,  298,  302,  307,  311,  322,
        328,  332,  374,  403,  405,  412,  426,  441,  445,  446,  458,
        460,  481,  587,  625,  669,  696,  730,  831,  839,  887,  909,
        911, 1034, 1040, 1069, 1081, 1088, 1098, 1116, 1183, 1235, 1246,
        1263, 1334, 1355, 1363, 1447, 1455, 1502, 1553, 1578, 1594, 1624,
        1723, 1737, 1743, 1841, 1867, 1898, 1926, 1963, 1970, 1992, 2002,
        2007, 2092, 2136, 2159, 2182, 2196, 2289, 2384, 2391, 2416, 2436,
        2514, 2518, 2691, 2764, 2817, 2836, 2843, 2893, 2910, 2911, 2952,
        2989, 3112, 3168, 3214, 3235, 3240, 3242, 3259, 3374, 3436, 3440,   
        3486, 3526, 3607, 3664, 3699, 3723, 3807, 3830, 3857, 3879, 3947,
        3982, 3983, 4014, 4018, 4051, 4088, 4137, 4160, 4195, 4214, 4267,
        4297, 4340, 4362, 4397, 4406, 4441, 4481, 4640, 4680, 4788, 4813,
        4830, 4853, 4859, 4898, 5022, 5049, 5104, 5163, 5192, 5322, 5339,
        5390, 5393, 5456, 5463, 5514, 5561, 5652, 5678, 5688, 5703, 5750,
        5778, 5789, 5808, 5867, 6000, 6019, 6064, 6078, 6081, 6147, 6181,
        6209, 6272, 6367, 6385, 6415, 6437, 6454, 6476, 6529, 6531, 6563,
        6818, 6836, 6848, 6880, 6925, 7059, 7067, 7078, 7113, 7148, 7178,
        7190, 7226, 7264, 7278, 7302, 7312, 7367, 7402, 7447, 7505, 7511,
        7517, 7635, 7780, 7794, 7800, 7859, 8014, 8051, 8063, 8088, 8095,
        8098, 8108, 8123, 8226, 8238, 8312, 8324, 8419, 8425, 8465, 8468,
        8580, 8609, 8629, 8630, 8747, 8770, 8797, 8838, 8975,
        1272, 1462, 1673,  174, 1919, 1988, 1993, 2035, 2078, 2086, 2277,
        2412, 2428,  251, 2803, 2902, 3000, 3081, 3170, 3536, 3576, 3752,
        3853,  422, 5338, 5536, 5694, 5895, 6241, 6295, 6313, 6319, 6345,
        652,  777, 7850, 7976, 8297,   84, 8842,
        1089, 1188,  121, 1221, 1284, 1320, 1580, 1995, 2094, 2300,  237,
        260, 2830, 2961, 3570, 3575, 3729, 4077, 4446, 4507, 4970, 4992,
        5105, 5142, 5639, 5683,   61,  672, 6829, 6930, 7021, 7127, 7176,
        7729, 8224, 8230, 8455, 8463, 8555,  908}
   sorted_list=sorted(list_speakres, key=lambda x: int(x))
   new_id=50365+sorted_list.index(speaker_id)
   return new_id


def cos_score(x: np.ndarray, y: np.ndarray):
    """Compute the cosine score between matrices x and y efficiently,
    where x is shape (N_train, N_embed) and y is shape (N_test, N_embed)"""

#    assert x.shape[1] == y.shape[1], "Embedding dimension must match"
    xn = np.sqrt((x * x).sum(axis=1, keepdims=True))
    yn = np.sqrt((y * y).sum(axis=1, keepdims=True))
    return np.dot(x, y.transpose()) / xn / yn.transpose()


def Processor(model_size:str):

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", language="English", task="transcribe")
    return processor

           
def loader(path_train:str,path_valid:str,path_dev:str,path_test:str,path_audio:str,batch_size:int,model_size:str):

    train_dataset  = LibriSpeechDataset(csv_file=path_train, audio_path=path_audio)
    val_dataset    = LibriSpeechDataset(csv_file=path_valid, audio_path=path_audio)
    dev_dataset    = LibriSpeechDataset(csv_file=path_dev, audio_path=path_audio)
    test_data      = LibriSpeechDataset(csv_file=path_test, audio_path=path_audio)
    _,processor    = base_model(model_size,Config.only_decoder)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator,num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.evalbatch_size, shuffle=False, collate_fn=data_collator,num_workers=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=Config.evalbatch_size, shuffle=False, collate_fn=data_collator,num_workers=1)
    test_dataloader = DataLoader(test_data, batch_size=Config.evalbatch_size, shuffle=False, collate_fn=data_collator,num_workers=1)

    return train_dataloader , val_dataloader, dev_dataloader,test_dataloader


class LibriSpeechDataset(Dataset):

    def __init__(self, csv_file, audio_path):
        self.data = pd.read_csv(csv_file)
        self.audio_path = audio_path
        _,self.processor = base_model(Config.model_size,Config.only_decoder)
        self.update_id=apdateID
#        self.normalize=normalize_text
    def __len__(self):
        return  len(self.data)

    def __getitem__(self, idx):
        audio_file = self.data.iloc[idx]['path']
        audio, sr =  librosa.load(audio_file, sr=16000,mono=True)
        speaker_id =  self.data.iloc[idx]['speaker_id']
        book_id =  self.data.iloc[idx]['book_id']
        utterance_id =  self.data.iloc[idx]['utterance_id']
        transcription =  self.data.iloc[idx]['transcription']

        assert(len(audio)/sr<30)       
        updated_id=self.update_id(speaker_id )
        tobeappended=[50258,50259, 50359,50363,updated_id]
        input_feature =  self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0] 
#        skript=self.processor.tokenizer._normalize(transcription) 
        labels =  self.processor.tokenizer(transcription).input_ids
#        labels =  self.processor.tokenizer(skript).input_ids
        labels = tobeappended + labels[4:]
        assert(len(labels)<448)
        return {"audio":audio,"input_features":input_feature,"labels":labels ,"text": transcription,"speaker_id":updated_id}



