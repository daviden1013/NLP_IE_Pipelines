# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.optim as optim
from modules.Utilities import Information_Extraction_Document
from modules.RE_utilities import InlineTag_RE_Dataset, RE_Trainer
import logging


def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('Training pipeline started')
  """ load config """
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
  
  logging.info('Config loaded:')
  pprint.pprint(config, sort_dicts=False)
  """ Load development id """
  logging.info('Creating datasets...')
  with open(config['deve_id_file']) as f:
    lines = f.readlines()
  dev_id = [line.strip() for line in lines]
  """ Load label_map """
  label_map = config['label_map']
  """ Load tokenizer """
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  """ Train-valid split """
  valid_ids = np.random.choice(dev_id, int(len(dev_id) * config['valid_ratio']), replace=False).tolist()
  train_ids = [i for i in dev_id if i not in valid_ids]
  """ Load IEs for dataset """
  train_IEs = {}
  for train_id in train_ids:
    train_IEs[train_id] = Information_Extraction_Document(doc_id=train_id, 
                                                          filename=os.path.join(config['IE_dir'], f'{train_id}.ie'))
      
  valid_IEs = {}
  for valid_id in valid_ids:
    valid_IEs[valid_id] = Information_Extraction_Document(doc_id=valid_id, 
                                                          filename=os.path.join(config['IE_dir'], f'{valid_id}.ie'))
  
  train_dataset = InlineTag_RE_Dataset(IEs=train_IEs, 
                                       tokenizer=tokenizer, 
                                       possible_rel=config['possible_rel'],
                                       token_length=config['token_length'], 
                                       label_map=label_map,                
                                       has_label=True)
  
  valid_dataset = InlineTag_RE_Dataset(IEs=valid_IEs, 
                                       tokenizer=tokenizer, 
                                       possible_rel=config['possible_rel'],
                                       token_length=config['token_length'], 
                                       label_map=label_map,                
                                       has_label=True)
  
  logging.info('Training dataset created')
  """ define model """
  logging.info(f"Loading base model from {config['base_model']}...")
  model = AutoModelForSequenceClassification.from_pretrained(config['base_model'], num_labels=len(label_map))
  logging.info('Model loaded')
  """ Load optimizer """
  optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
  """ Training """
  trainer = RE_Trainer(run_name=config['run_name'], 
                        model=model,
                        n_epochs=config['n_epochs'],
                        train_dataset=train_dataset,
                        batch_size=config['batch_size'],
                        optimizer=optimizer,
                        valid_dataset=valid_dataset,
                        save_model_mode='best',
                        save_model_path=os.path.join(config['out_path'], 'checkpoints'),
                        log_path=os.path.join(config['out_path'], 'logs'))
  
  trainer.train()

if __name__ == '__main__':
  main()