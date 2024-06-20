import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor
from pytorch_lightning.tuner  import Tuner
from model.WhisperLight import WhisperModel,base_model
from functools import partial
from Libdata import loader
from config import Config


device = 'cuda' if torch.cuda.is_available  else 'cpu'



def train():
    device = 'cuda' if torch.cuda.is_available  else 'cpu'
    pl.seed_everything(42)
    cuda_available = torch.cuda.is_available()
    _,processor = base_model(Config.model_size,Config.only_decoder)
    train_dataloader,val_dataloader,dev_dataloader,_= loader(Config.train_data,Config.validation_data,
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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="whisper-{epoch:02d}.step_{step:06d}-val-loss_{val_loss:.4f}",
        mode="min",
        save_top_k=-1)

    trainer = pl.Trainer( 
        default_root_dir ="/vol/csedu-nobackup/project/fharrathi/Base",
#        precision        =16-mixed,
        max_steps        =Config.training_steps,
#        max_epochs=2,
        accelerator = device,
        accumulate_grad_batches=Config.gradient_steps,
        callbacks=[checkpoint_callback,EarlyStopping(monitor="val_loss", mode="min",patience=2)])

    trainer.fit(model=module,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

if __name__ == "__main__":

      train()
