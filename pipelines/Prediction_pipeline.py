# -*- coding: utf-8 -*-
import os
import argparse
from easydict import EasyDict
import yaml
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from modules.Utilities import Information_Extraction_Document
from modules.NER_utilities import Sentence_NER_Dataset, NER_Predictor
from modules.RE_utilities import InlineTag_RE_Dataset, RE_Predictor
import logging
import pprint
import pandas as pd


def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('Prediction pipeline starts:')
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))

  logging.info('Config loaded:')
  pprint.pprint(config, sort_dicts=False)
  
  """ Convert documents into a list of IEs """
  df = pd.read_pickle(config['doc_file'])
  input_IEs = []
  for i, r in df.iterrows():
    input_IEs.append(Information_Extraction_Document(doc_id=r[config['doc_id_var']], text=r[config['text_var']]))
  """
  Named Entity Recognition
  """
  """ Load NER label map """
  ner_label_map = config['NER_label_map']
  """ load tokenizer """
  tokenizer = AutoTokenizer.from_pretrained(config['NER_tokenizer'])
  """ NER Dataset """    
  ner_dataset = Sentence_NER_Dataset(IEs=input_IEs, 
                                     tokenizer=tokenizer, 
                                     label_map=ner_label_map, 
                                     token_length=config['NER_token_length'], 
                                     has_label=False,
                                     mode=config['BIO_mode'])
  
  logging.info('Datasets created')
  """ load model """
  logging.info('Loading NER model...')
  ner_model = AutoModelForTokenClassification.from_pretrained(config['NER_model'])
  """ Prediction """
  logging.info('NER Predicting...')
  predictor = NER_Predictor(model=ner_model,
                            tokenizer=tokenizer,
                            dataset=ner_dataset,
                            label_map=ner_label_map,
                            batch_size=config['batch_size'],
                            device=config['device'])
  
  ner_IEs = predictor.predict()
  logging.info('NER IEs predicted')
  """
  Relation Extraction
  """
  """ Load label map """
  re_label_map = config['re_label_map']
  """ load tokenizer """
  tokenizer = AutoTokenizer.from_pretrained(config['RE_tokenizer'])
  """ RE Dataset """  
  re_dataset = InlineTag_RE_Dataset(IEs=ner_IEs, 
                                    tokenizer=tokenizer, 
                                    possible_rel=config['possible_rel'],
                                    token_length=config['RE_token_length'], 
                                    label_map=re_label_map, 
                                    has_label=False)
  logging.info('Datasets created')
  """ load model """
  logging.info('Loading model...')
  re_model = AutoModelForSequenceClassification.from_pretrained(config['RE_model'])
  logging.info('Model loaded')
  """ Prediction """
  logging.info('RE Predicting...')
  predictor = RE_Predictor(model=re_model,
                            tokenizer=tokenizer,
                            dataset=re_dataset,
                            label_map=re_label_map,
                            batch_size=config['batch_size'],
                            device=config['device'])
  
  pred_IEs = predictor.predict()  
  logging.info('IEs predicted')
  
  """ Save output IEs """
  for ie in pred_IEs:
    ie.save(os.path.join(config['out_dir'], f"{ie['doc_id']}.ie"))
  
if __name__ == "__main__":
  main()