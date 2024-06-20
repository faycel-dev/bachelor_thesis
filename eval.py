import torch
from model.WhisperLight import base_model,WhisperModel
from config import Config
from Libdata import loader
import os
import pytorch_lightning as pl


device = 'cuda' if torch.cuda.is_available  else 'cpu'
def find_best_checkpoint(version:str):
    cuda_available = torch.cuda.is_available()
    _,processor = base_model(Config.model_size,Config.only_decoder)
    train_dataloader,val_dataloader,dev_dataloader,test_dataloader= loader(Config.train_data,
                                                                           Config.validation_data,
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
    train_dataloader,val_dataloader,dev_dataloader,test_dataloader= loader(Config.train_data,Config.validation_data,
                                                                           Config.dev_data,Config.test_data ,
                                                                           Config.audio_path ,
                                                                           batch_size=Config.batch_size,
                                                                           model_size=Config.model_size)
    data_eval=[dev_dataloader,test_dataloader]
    path =Config.lightning_directory
    version=Config.version
    checkpoint_file=Config.best_ckpt
    checkpoint=f"{path}/{version}/checkpoints/{checkpoint_file}"
    trainer = pl.Trainer(accelerator = device)
    final_result=[]
    for data in data_eval:
       print(checkpoint)
       model=WhisperModel.load_from_checkpoint(checkpoint,max_steps=Config.training_steps)
       model.eval()
       results=trainer.test(model,dataloaders=data)
#           results["data"]=f"{data}"
#           results["checkpoint"]=f"reslults for checkpoint {checkpoint} : {results}"
#       final_result.append(results)
       print(final_result)




if __name__=="__main__":

      #find_best_checkpoint(Config.version)
#      find_best_checkpoint(Config.version)
     evaluate_Best_checkpoint()
