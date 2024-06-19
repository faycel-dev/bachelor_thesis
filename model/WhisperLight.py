import torch
import pytorch_lightning as pl
from transformers import( WhisperForConditionalGeneration,
                          WhisperProcessor)
import jiwer as jw
import torch.nn as nn
import torch.optim as optim
from typing import Optional,Tuple
from model.ThreeStage import ThreeStage
from config import Config
from  compute_err import compeer 
import re
#import csv


def base_model(model_size:str,only_decoder:bool):
    """
    Load a pre-trained Whisper model and its associated processor.
    Args:
    model_size (str): Size of the Whisper model to load,  eg: tiny,base ,small or medium.    """
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", language="english", task="transcribe")
    model=WhisperForConditionalGeneration.from_pretrained(f'openai/whisper-{model_size}')  

    if only_decoder:
       model.freeze_encoder()
    return model, processor

class WhisperModel(pl.LightningModule):

    def __init__(self,max_steps:int):
        super().__init__()
        self.save_hyperparameters
     
        self.model ,self.processor=base_model(Config.model_size,Config.only_decoder)
        self.model.config.dropout=Config.dropout
        self.lr = Config.max_learning_rate
        threestage = ThreeStage(max_steps, [1, 0, 9]) 
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr) 
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, threestage.lr)


        #  metrics for computing WER and EER
        self.metric= jw.wer
        self.metric2 = compeer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.speaker_index =3        #index where we extract the speaker embedding from the decoder hidden states 
        # used in case we want to compute the EER for all decder layers
#        self.embeddings=[[] for _ in range(self.model.config.num_hidden_layers +1)]
        self.embeddings=[]
        self.speaker_ids=[]
        self.references   =[]        #store the references
        self.hypothesis   =[]        #store the predictions
        self.counter =0              #number of speakers in the validations set

    def forward(self ,batch):
         output= self.model(batch['input_features'],labels=batch['labels'],output_hidden_states=True)
         return output


    def training_step(self, batch, batch_idx):
        # compute loss, logits and hidden_states by a forward pass 
        output=self.forward(batch)
        loss=output['loss']
        logits=output['logits']
        #get the hypothesis by applying softmax and taging the highest probability for each token  
        predictions=output.logits.softmax(-1).argmax(-1)
        labels=batch['labels']
        # remove non necessary tokens for computing WER
        prediction,label=self.process(labels,predictions)  
        # decode 
        pridicted_labels=self.processor.tokenizer.batch_decode(prediction, skip_special_tokens=True)
        true_label=self.processor.tokenizer.batch_decode(label, skip_special_tokens=True)
      
        wer= self.metric(true_label,pridicted_labels)
        self.log("train_wer",wer,on_step=False,on_epoch=True)

        # Log the training loss
        self.log("learning_rate",self.optimizer.param_groups[0]['lr'])
        self.log("train_loss", loss, on_step=True,on_epoch=True, logger=True) #,prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output=self.forward(batch)
        predictions=output.logits.softmax(-1).argmax(-1)
        labels=batch['labels']
        prediction,label=self.process(labels,predictions)        
        pridicted_labels=self.processor.tokenizer.batch_decode(prediction, skip_special_tokens=True)
        true_label=self.processor.tokenizer.batch_decode(label, skip_special_tokens=True)
        wer= self.metric(true_label,pridicted_labels)
        self.log("val_wer",wer,on_step=False,on_epoch=True)
        loss=output['loss']
       # speaker_embedding=output.decoder_hidden_states[-1][0,3]
       # speaker_id=batch['speaker_ids']
        print(self.model.proj_out(output.decoder_hidden_states[-1])[0,self.speaker_index,:].softmax(-1).argmax(-1).item())
        print(batch["speaker_ids"].item())
        # compute the correctly predicted speaker ID tokens 
        if self.model.proj_out(output.decoder_hidden_states[-1])[0,self.speaker_index,:].softmax(-1).argmax(-1).item()==batch["speaker_ids"].item(): #  50363:
            self.counter+=1
        self.log("val_loss", loss, on_step=False,on_epoch=True,logger=True,prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Log the counter value at the end of the epoch
        self.log("counter", self.counter, on_epoch=True, logger=True, prog_bar=True)
        #reset the counter
        self.counter=0
        return


    def test_step(self, batch, batch_idx):
        output = self.forward(batch) 
        input_features=batch['input_features']
        labels=batch['labels']

        self.speaker_ids.append(batch["speaker_ids"])
        # used to store the speaker embeddings from all layers

        #for layer_idx in range(self.model.config.num_hidden_layers+1):
        #   self.embeddings[layer_idx].append(output.decoder_hidden_states[layer_idx][0,self.speaker_index,:])
 
#        mean=torch.mean(output.decoder_hidden_states[self.model.config.num_hidden_layers].squeeze(0),dim=0).tolist()
#        self.avembeddings.append(mean)

        self.embeddings.append(output.decoder_hidden_states[-1][0,self.speaker_index,:].tolist())
        _,truth_label,prediction=self.compute_WER(input_features,labels)
        self.hypothesis.append(prediction)
        self.references.append(truth_label)
#        test_wer,_=self.compute_WER(input_features,labels)
        loss=output.loss
        #self.log("test_wer",test_wer,on_step=False,on_epoch=True)
 #       self.log('test_loss', loss)  # Log the test loss for visualization
        return loss     

    def on_test_epoch_end(self):

         speaker_ids = [x.item() for x in self.speaker_ids]

#         for layer_idx in range(self.model.config.num_hidden_layers +1):
#              embeddings = [x.tolist() for x in self.embeddings[layer_idx]]
#              eer_val = self.metric2(embeddings, speaker_ids)
#              self.log(f"eer_val_layer_{layer_idx}", eer_val)

         embeddings = [x for x in self.embeddings]
         eer_val = self.metric2(embeddings,speaker_ids)
         self.log("val_val",eer_val)
        
         normpredictions = [v[0] for v in self.hypothesis]
         normtruth_labels =[v[0] for v in self.references]
         
         

         # compute WER
         norm_wer=100*self.metric(normtruth_labels,normpredictions)
         self.log("normalized_wer",norm_wer)


    def configure_optimizers(self):
        scheduler_config = dict(scheduler=self.scheduler, interval="step", strict=False, name='threestage')
        return dict(optimizer=self.optimizer, lr_scheduler=scheduler_config)

    def compute_WER(self,input_features, labels):
 
        """
        computes the word error rate 
        input batch (dict): A dictionary containing input features and labels .
        output : word error rate together with truth_labels (references) and  predictions (hypothesis) in form of text 
                       to analyse the type of error         
        """
        label=labels.clone()
        # generate the predicted tokens using generate function
        predected_tokens=self.model.generate(input_features,language='en')
        # remove speaker ID token and replace it with the padding token
        predected_tokens=torch.where(predected_tokens <= 50363,predected_tokens,self.processor.tokenizer.pad_token_id)
        #remove the -100  tokens and replace them with the padding token
        label[label == -100] = self.processor.tokenizer.pad_token_id
        label=torch.where(label <= 50363,label,self.processor.tokenizer.pad_token_id)
       
        prediction = self.processor.tokenizer.batch_decode(predected_tokens, skip_special_tokens=True)
        truth_label = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions = [self.processor.tokenizer._normalize(v) for v in prediction ]
        truth_labels = [self.processor.tokenizer._normalize(v) for v in truth_label]

        wer_score =100*self.metric(truth_labels,predictions)
        #unormalized_wer=100*self.metric(truth_label,prediction)
        return wer_score,truth_labels,predictions
    
    def process(self,predictions,labels):
           #remove the -100 tokens
           labels[labels == -100] = self.processor.tokenizer.pad_token_id
           #remove the speaker id
           label=torch.where(labels <= 50363,labels,self.processor.tokenizer.pad_token_id)
           # replace the speaker ID token  by the padding token
           prediction=torch.where(predictions <= 50363,predictions,self.processor.tokenizer.pad_token_id)
           return prediction, label


