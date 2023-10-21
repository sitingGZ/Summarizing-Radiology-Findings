from posixpath import dirname
from typing import List, Dict
import os
#import logging
#from logging import Logger
#import queue
#from random import shuffle
import numpy as np
from pytorch_lightning.core.hooks import CheckpointHooks
import torch
import pytorch_lightning as pl
from joeynmt.helpers import load_config, set_seed, make_model_dir, load_checkpoint, symlink_update
#from joeynmt.builders import build_gradient_clipper
#from transformers import AutoModel

from vocabulary import TransformersLexicon
from data import ReportData, make_train_corpus
from pointer_model.Seq2SeqDataset import SummaryData
from pointer_model.EncoderDecoderModels import Bert2BertModel

from torch.utils.data import DataLoader as batch_gen
from torch import autograd
#from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(config_file):
    # Prepare configurations
    configs = load_config(config_file)
    set_seed(seed=configs['train'].get('random_seed', 42))
    
    batch_size = configs['train']['batch_size']
    encoder_name = configs['model']['encoder']['name']
    decoder_name = configs['model']['decoder']['name']
    #devices = [configs['model']['encoder']['device'], configs['model']['decoder']['device']]
    #gpus = len(set(devices))
    gpus = 1
    #gpus = 0
    max_epochs = configs['train']['epochs']
    # Get training data and pre-trained language model
    tsv_path = configs['data']['tsv']
    data_loader = ReportData(tsv_path)
    train_data, valid_data, test_data = data_loader.get_middle_range_test_data()
    #train_dataset = SummaryData(train_data,  sep_token='[CLS]', select_random=False)
    #valid_dataset = SummaryData(valid_data,  sep_token='[CLS]', select_random=False)
    train_dataset = SummaryData(train_data, sep_token= ' ', eos_token='<|endoftext|>' )
    valid_dataset = SummaryData(valid_data, sep_token= ' ', eos_token='<|endoftext|>' )
    if 'vocab_weight' in configs['train']:
        vocab_weight = np.array([float(l.strip().split('\t')[-1]) for l in open (configs['train']['vocab_weight'])])
        vocab_weight = torch.FloatTensor(vocab_weight)
    else:
        vocab_weight = None

    # Initial EncoderDecoderModel with german bert with maximum positions
    pointer_ratio = configs['model']['pointer_ratio']
    
    
    #for pointer_ratio in [0.9, 0.8, 0.7, 0.6]:
    for reconstruct in [False, True]:
    #for coverage_weight in [0.5, 1.0]:
        if reconstruct:
            batch_size = 2
            max_epochs = 12
        
        train_loader = batch_gen(train_dataset, batch_size, num_workers=2)
        valid_loader = batch_gen(valid_dataset, batch_size, num_workers=2)
        
        model_dir = configs['train']['model_dir'].format(reconstruct, pointer_ratio)
        bert2bert = Bert2BertModel(config_file, reconstruct = reconstruct, pointer_ratio = pointer_ratio, vocab_weight=vocab_weight)

        if "best_checkpoint" in configs['train'] and os.path.exists(configs['train']['best_checkpoint']):
            #bert2bert.load_from_checkpoint(configs['train']['best_checkpoint'])
            checkpoints = torch.load(configs['train']['best_checkpoint'])
            bert2bert.load_state_dict(checkpoints['state_dict'])
            print('Loaded best checkpoint: {}'.format(configs['train']['best_checkpoint']))
        else:
            bert2bert._load_pretrained_weights(encoder_name, decoder_name)
            print('Loaded weights for encoder {} and decoder {}. '.format(encoder_name, decoder_name))

        #print('Devices after Loaded weights for encoder {} and decoder {}. '.format(bert2bert.model.encoder.device, bert2bert.model.decoder.device))
    
        torch.cuda.empty_cache()
        checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss', dirpath=model_dir, mode = 'min',
                                                    filename='bert2bert-{epoch:02d}-{val_loss:.5f}', save_top_k=2)
                                  
        # trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
        trainer = Trainer(gpus=gpus, gradient_clip_val = 1.0, stochastic_weight_avg=True, max_epochs=max_epochs,callbacks=checkpoint_callback, precision=16)
        trainer.fit(bert2bert, train_loader, valid_loader)
        print('Best model path', checkpoint_callback.best_model_path)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 2, "The path to the config file must be given as argument!"
    main(sys.argv[1])
    #import torch
    #Checkpoint = "models/bert2bert_no_pointer/bert2bert-epoch=04-val_loss=1.19.ckpt"
    #checkpoints = torch.load(Checkpoint)
    #print(checkpoints.keys(), checkpoints['state_dict'].keys())