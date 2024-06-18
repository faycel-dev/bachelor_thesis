#!/usr/bin/env python3

class Config():
#   def __init__(self):
        # Model configuration
        model_size: str = 'base'
        # Training configurati
        batch_size     : int  = 2
        max_learning_rate  : float= 3e-5
        training_steps : int  = 6000  # Add the training_steps attribute
        gradient_steps : int  = 8
        evalbatch_size : int  = 1
        dropout               = 0.0

        only_decoder:bool =True
      
        #Twostage parameter
        warmup_steps: int =600
        min_lr =2e-6
        final_lr =8e-6
        # Add more training parameters as needed
        path_train  = '/scratch/fharrathi/data/LibriSpeech/train_dataset.csv'
        path_valid  = '/scratch/fharrathi/data/LibriSpeech/val_dataset.csv'
        audio_path  = '/scratch/fharrathi/data/LibriSpeech/'
        test_data   = '/scratch/fharrathi/data/LibriSpeech/clean_test.csv'
        dev_data   = '/scratch/fharrathi/data/LibriSpeech/dev_cleann.csv'

