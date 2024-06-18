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
from WhisperClean5 import WhisperModel,base_model
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
        self.normalize=normalize_text
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


def evaluate():
    device = 'cuda:0' if torch.cuda.is_available  else 'cpu'
    data = LibriSpeechDataset(csv_file=Config.dev_data, audio_path=Config.audio_path)
    #state_dict = torch.load('./models/SmallW1/whisper-epoch=02-English-smal1305-val_loss=0.20.ckpt') #, map_location=torch.device('cpu'))
    #state_dict = state_dict['state_dict']
#    norm=EnglishTextNormalizer() 
#    model=whisper.load_model("medium.en")
    model,processor = base_model(Config.model_size,Config.only_decoder)
#    module=WhisperModel.load_from_checkpoint("/vol/csedu-nobackup/project/fharrathi/Small/lightning_logs/version_4455717/checkpoints/whisper-epoch=01.step_step=002842-val-loss_val_loss=0.0803.ckpt",max_steps=Config.training_steps)
#    module.eval()
   # model.eval()
    model.to(device)

    ref=[]
    pred=[]
    j=0
    for i in tqdm(range(len(data))):
#       ref.append(normalize_text(data[i]["text"]))
       ref.append(processor.tokenizer._normalize(data[i]["text"]))
      # ref.append(norm(data[i]["text"]))

#       print("====>>     {}".format(data[i]["text"]))
      # pred.append(norm(model.transcribe(data[i]['audio'])['text']))
       input_features=processor.feature_extractor(data[i]['audio'],sampling_rate=16000, return_tensors="pt").input_features
       with torch.no_grad():
            prediction=model.generate(input_features.to(device),language="en",task="transcribe")
       #predictions=torch.where(prediction <= 50363,prediction,processor.tokenizer.pad_token_id)
#       print("the prediction is {}".format(prediction))
       clean=processor.tokenizer._normalize(processor.tokenizer.decode(prediction[0],skip_special_tokens=True))
    #   clean=normalize_text(processor.tokenizer.decode(prediction[0],skip_special_tokens=True))

       pred.append(clean)

    for i ,j in zip(ref,pred):
          print(i)
          print(j)
          print("#############")
    print("the word error rate , model fine tun base test ======> {}".format(100 * jw.wer(ref ,pred))) 
 
     

def eer_layer(layer:int):
    test_dataset = LibriSpeechDataset(csv_file=Config.test_data, audio_path=Config.audio_path)
    dev_dataset = LibriSpeechDataset(csv_file=Config.dev_data, audio_path=Config.audio_path)

    counter=0
    _,processor = base_model(Config.model_size,Config.only_decoder)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator, num_workers=1)  # Adjust batch size as per available memory
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=data_collator, num_workers=1)  # Adjust batch size as per available memory

    module=WhisperModel.load_from_checkpoint("/vol/csedu-nobackup/project/fharrathi/Base/lightning_logs/version_4459435/checkpoints/whisper-epoch=02.step_step=004263-val-loss_val_loss=0.1130.ckpt",
                                             max_steps=Config.training_steps)
#    module.model.eval()
    embeddings = []    
    speakers=[]
    counter=0
#    print("dev_data")
#    print(module.model.generation_config)
    for ind, row in enumerate(test_dataloader):
            input_features = row['input_features'].to(device)
            labels = row['labels'].to(device)
            speaker=row['speaker_ids'].to(device)
            output = module.model(input_features, labels=labels, output_hidden_states=True)
            last_hidden_state=output.decoder_hidden_states[layer]
            embeddings.append(last_hidden_state[0,3,:].tolist())
            logits_position =module.model.proj_out(last_hidden_state)
            if logits_position[0,3].softmax(-1).argmax(-1).item() > 50363:
               counter+=1
               print(logits_position[0,3].softmax(-1).argmax(-1).item() )
            else:
               print("missed {} at index {}".format(labels,ind))
               print(logits_position[0,3,:].softmax(-1).argmax(-1).item())
               print(module.model.generate(input_features,language="en"))
               print(module.model.proj_out(last_hidden_state).softmax(-1).argmax(-1))
            speakers.append(speaker.item())
    print("counter = {}".format(counter))   
    embeddings = np.array(embeddings)
    speakers = np.array(speakers)
#    print("Embeddings shape", embeddings.shape)
    scores = cos_score(embeddings, embeddings)
    labels = np.expand_dims(speakers, 1) == speakers
    assert scores.shape == labels.shape
#    print("Scores shape", scores.shape)
    upper_tri = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    flat_scores = scores[upper_tri]
    flat_labels = labels[upper_tri]
#    print("Flat scores shape", flat_scores.shape)
    print(f"Equal Error Rate = {eer.eer(flat_scores, flat_labels):4.1%}")
    return eer.eer(flat_scores, flat_labels)
    
def eer_alllayers(number_layer: int):
     print("test dataset")
     for i in range(number_layer +1):
         print("layer  ...........",i)
         error_rate=eer_layer(i)
         print(f"Equal Error rate for layer {i} = {error_rate:4.1%}")

def find_best_checkpoint(version:str):
    cuda_available = torch.cuda.is_available()
    _,processor = base_model(Config.model_size,Config.only_decoder)
    train_dataloader,val_dataloader,dev_dataloader,test_dataloader= loader(Config.path_train,
                                                                           Config.path_valid,
                                                                           Config.dev_data,
                                                                           Config.test_data ,
                                                                           Config.audio_path ,
                                                                           batch_size=Config.batch_size,
                                                                           model_size=Config.model_size)
    path1 ="/vol/csedu-nobackup/project/fharrathi/Small/lightning_logs"
    checkpoint_dir = os.path.join(path1, version,"checkpoints")
    checkpoint_files = os.listdir(checkpoint_dir)
    data_eval=[dev_dataloader,test_dataloader]
    checkpoint_files = [file for file in checkpoint_files if file.endswith(".ckpt")]
    trainer = pl.Trainer(accelerator = device)
    print(checkpoint_files)
    final_result=[]
    for data in data_eval:     
        for checkpoint_file in checkpoint_files:
            checkpoint=f"{path1}/{version}/checkpoints/{checkpoint_file}"
            print(checkpoint)
            model=WhisperModel.load_from_checkpoint(checkpoint,max_steps=Config.training_steps)
            model.eval()
            results=trainer.test(model,dataloaders=data)
#        results["checkpoint"]=int(version.split('_')[-1])
            final_result.append(results)
#        print(final_result)
    return final_result


def evaluate_Best_checkpoint():
    cuda_available = torch.cuda.is_available()
    _,processor = base_model(Config.model_size,Config.only_decoder)
    train_dataloader,val_dataloader,dev_dataloader,test_dataloader= loader(Config.path_train,Config.path_valid,
                                                                           Config.dev_data,Config.test_data ,
                                                                           Config.audio_path ,
                                                                           batch_size=Config.batch_size,
                                                                           model_size=Config.model_size)
    data_eval=[dev_dataloader,test_dataloader]
    path1 ="/vol/csedu-nobackup/project/fharrathi/Small/lightning_logs"
    version="version_4497221"
    checkpoint_file="whisperDEC-epoch=01.step_step=002842-val-loss_val_loss=0.0738.ckpt"
    checkpoint=f"{path1}/{version}/checkpoints/{checkpoint_file}"
    trainer = pl.Trainer(accelerator = device)
    final_result=[]
    for data in data_eval:
       print(f"evaluating data....")
       #for checkpoint_file in checkpoint_files:
       print(checkpoint)
       model=WhisperModel.load_from_checkpoint(checkpoint,max_steps=Config.training_steps)
       model.eval()
       results=trainer.test(model,dataloaders=data)
#           results["data"]=f"{data}"
#           results["checkpoint"]=f"reslults for checkpoint {checkpoint} : {results}"
#       final_result.append(results)
       print(final_result)

def main2():
    device = 'cuda' if torch.cuda.is_available  else 'cpu'
    pl.seed_everything(42)
    cuda_available = torch.cuda.is_available()
    _,processor = base_model(Config.model_size,Config.only_decoder)
    train_dataloader,val_dataloader,dev_dataloader,_= loader(Config.path_train,Config.path_valid,
                                                             Config.dev_data,Config.test_data ,Config.audio_path ,
                                                             batch_size=Config.batch_size,
                                                             model_size=Config.model_size)
    module= WhisperModel(Config.training_steps)
    #module.model.config.forced_decoder_ids =None
    #module.model.config.use_cache=False
#    module.model.generation_config.forced_decoder_ids=[]
    module.model.config.use_cashe=False
    module.model.generate = partial(module.model.generate, language="english", task="transcribe", use_cache=True )
#    module.model.config.apply_spec_augment=True
#    module.model.config.mask_feature_prob=0.1
    module.model.generation_config.language ="en"
#    module.model.freeze_encoder()
#    module.model.train()
#    module.to(device)

# Create  PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="whisper-{epoch:02d}.step_{step:06d}-val-loss_{val_loss:.4f}",
        mode="min",
        save_top_k=-1)

    trainer = pl.Trainer( 
        default_root_dir ="/vol/csedu-nobackup/project/fharrathi/Base",
#        precision        =16-mixed,
        max_steps        =Config.training_steps,
#        log_every_n_steps=500,
#        max_epochs=2,
        accelerator = device,
        accumulate_grad_batches=Config.gradient_steps,
        callbacks=[checkpoint_callback,EarlyStopping(monitor="val_loss", mode="min",patience=2)])

    trainer.fit(model=module,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

if __name__ == "__main__":
    main2()
#    eer_alllayers(24)
#    evaluate()
#    eer_layer(-1)
#    evv()
#    x=find_best_checkpoint("version_4519233")
#    evaluate_Best_checkpoint()    
