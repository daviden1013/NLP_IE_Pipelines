# -*- coding: utf-8 -*-
import abc
import warnings
from typing import List, Dict, Tuple, Union
from itertools import combinations
import random
import os
from tqdm import tqdm
import yaml
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification


class Information_Extraction_Document:
  def __init__(self, 
               doc_id:str,
               filename:str=None,
               text:str=None, 
               entity_list:List[Dict[str, str]]=None,
               relation_list:List[Dict[str, str]]=None):
    """
    Information Extraction Document (IE) is a general object for NER and RE
    
    Parameters
    ----------
    doc_id : str
      Document ID. Must be a string.
    filename : str, Optional
      file (.ie) path to load. If provided, the text, entity_list, relation_list
      parameters will not be used. 
    text : str, Optional
      document text.
    entity_list : List[Dict[str, str, str, int, int]], Optional
      List of dict of {entity_id, entity_text, entity_type, start, end}.
    relation_list : List[Dict[str, str, str, str, str, str]], Optional
      List of dict of {relation_id, entity_1_id, entity_1_text, entity_2_id, 
                       entity_2_text, relation_type}.
    """
    assert filename or text, "A filename or a text must be provided."
    assert isinstance(doc_id, str), "doc_id must be a string."
    self.doc_id = doc_id
    # if create object from file
    if filename:
      with open(filename) as yaml_file:
        ie = yaml.safe_load(yaml_file)
      if 'text' in ie.keys():
        self.text = ie['text']
      if 'entity' in ie.keys():
        self.entity = ie['entity']
      if 'relation' in ie.keys():
        self.relation = ie['relation']
    # create object from raw inputs
    else:
      self.text = text
      self.entity = entity_list if entity_list is not None else []
      self.relation = relation_list if relation_list is not None else []
  
  def get_entity_by_id(self, entity_id:str):
    """ 
    This method returns an entity as dict {entity_id, entity_text, entity_type, start, end}
    """
    for e in self.entity:
      if entity_id == e['entity_id']:
        return e
    raise ValueError(f"{entity_id} not found")
    
    
  def __getitem__(self, key):
    if key in ["doc_id", "text", "entity", "relation"]:
      return getattr(self, key)
    else:
      raise KeyError(f"'{key}' is not a valid key.")
    
  def has_entity(self) -> bool:
    return bool(self.entity)
    
  def has_relation(self) -> bool:
    return bool(self.relation)
  
  def __repr__(self, N_top_chars:int=100, N_top_items:int=2) -> str:
    text_to_print = self.text[0:N_top_chars]
    entity_to_print = self.entity[0:N_top_items]
    relation_to_print = self.relation[0:N_top_items]
    return ''.join((f'Information_Extraction_Document(doc_id="{self.doc_id}")\n',
            f'text="{text_to_print}", \n',
            f'entity={entity_to_print}, \n',
            f'relation={relation_to_print})'))

  def save(self, filename:str):
    with open(filename, 'w') as yaml_file:
      yaml.safe_dump({'doc_id':self.doc_id, 
                      'text':self.text, 
                      'entity':self.entity, 
                      'relation':self.relation}, 
                     yaml_file, sort_keys=False)
      yaml_file.flush()


class IE_converter:
  def __init__(self, IE_dir:str, overlap_entity_action:str='random', overlap_entity_rank:List[str]=None):
    """
    This class inputs a directory with annotation files, outputs IEs

    Parameters
    ----------
    ann_dir : str
      Directory of annotation files
    IE_dir : str
      Directory of IE files 
    overlap_entity_action : str
      Actions to take if overlapped entities were detected. Must be {'random', 'priority', 'keep'}
      If "random", one of the entities will be randomly dropped.
      If "priority", the overlap_entity_rank must be provided. 
      If "keep", both entities will be kept. A warning will be triggered.
    """
    self.IE_dir = IE_dir
    assert overlap_entity_action in {'random', 'priority', 'keep'}, \
      "overlap_entity_action must be {'random', 'priority', 'keep'}"
      
    if overlap_entity_rank == 'priority':
      assert isinstance(overlap_entity_rank, list), \
        'When overlap_entity_rank="priority", the overlap_entity_rank must be provided.'
    self.overlap_entity_action = overlap_entity_action
    self.overlap_entity_rank =overlap_entity_rank
    

  @abc.abstractmethod
  def _parse_text(self) -> str:
    """
    This method inputs annotation filename with dir
    outputs the text
    """
    return NotImplemented

  @abc.abstractmethod
  def _parse_entity(self) -> List[Dict[str, str]]:
    """
    This method inputs annotation filename with dir
    outputs list of dict {entity_id, entity_text, entity_type, start, end}
    """
    return NotImplemented
  
  
  def _overlap(self, e1_start, e1_end, e2_start, e2_end) -> bool:
    return not (e1_end <= e2_start or e1_start >= e2_end)
  
  def _check_overlap_entities(self, entities: List[Dict[str, str]]) -> bool:
    pairs = combinations(entities, 2)
    for entity1, entity2 in pairs:
      if self._overlap(entity1['start'], entity1['end'], entity2['start'], entity2['end']):
          return True
    return False
  
  
  def _random_sample_overlap_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    drop_entity_ids = []
    pairs = combinations(entities, 2)
    for entity1, entity2 in pairs:
      if self._overlap(entity1['start'], entity1['end'], entity2['start'], entity2['end']):
        # If overlap, randomly choose one to drop
        if bool(random.getrandbits(1)):
          drop_entity_ids.append(entity1['entity_id'])
        else:
          drop_entity_ids.append(entity2['entity_id'])

    return [e for e in entities if e['entity_id'] not in drop_entity_ids]
  
  
  def _prioritize_overlap_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rank = {e:i for i, e in enumerate(self.overlap_entity_rank)}
    drop_entity_ids = []
    pairs = combinations(entities, 2)
    for entity1, entity2 in pairs:
      if self._overlap(entity1['start'], entity1['end'], entity2['start'], entity2['end']):
        # If overlap, choose the higher priority (prior) one
        entity1_rank = rank[entity1['entity_type']]
        entity2_rank = rank[entity2['entity_type']]
        if entity1_rank > entity2_rank:
          drop_entity_ids.append(entity1['entity_id'])
        else:
          drop_entity_ids.append(entity2['entity_id'])

    return [e for e in entities if e['entity_id'] not in drop_entity_ids]
  
  
  @abc.abstractmethod
  def _parse_relation(self) -> List[Dict[str, str]]:
    """
    This method inputs annotation filename with dir
    outputs list of dict {relation_id, relation_type, entity_1_id, entity_1_text, 
                          entity_2_id, entity_2_text}
    """
    return NotImplemented
  
  @abc.abstractmethod
  def pop_IE(self):
    """
    This method populates input annotation files and save as [doc_id].ie files. 
    """
    return NotImplemented


class Label_studio_IE_converter(IE_converter):
  def __init__(self, doc_id_var:str, relation_map:Dict[List[str], str], 
               ann_file:str, IE_dir:str, overlap_entity_action:str='random', 
               overlap_entity_rank:List[str]=None):
    """
    This class inputs an annotation file, outputs IEs
    Parameters
    ----------
    txt_dir : str
      Directory of text files
    relation_map : Dict[List[str], str]
      a dictionary of {entity_type, entity_type) : label}. This is because 
      Label-studio doesn't allow specifing relation type. The label key must 
      match label_map in RE pipeline config. Mis-matching will not be populated
    ann_dir : str
      Directory of annotation files
    IE_dir : str
      Directory of IE files 
    """
    super().__init__(IE_dir, overlap_entity_action, overlap_entity_rank)
    self.doc_id_var = doc_id_var
    self.relation_map = {}
    for k, t in relation_map.items():
      self.relation_map[(t[0], t[1])] = k
      self.relation_map[(t[1], t[0])] = k
    
    self.ann_file = ann_file
    with open(self.ann_file, encoding='utf-8') as f:
      self.annotation = json.loads(f.read())
      
  
  def _parse_doc_id(self, idx:int) -> str:
    ann = self.annotation[idx]
    return ann['data'][self.doc_id_var]
  
  
  def _parse_text(self, idx:int) -> str:
    ann = self.annotation[idx]
    return ann['data']['text']
  
  
  def _parse_entity(self, idx:int) -> List[Dict[str, str]]:
    entity_list = []
    ann = self.annotation[idx]
    for r in ann['annotations'][0]['result']:
      if r['type']=='labels':
        entity_list.append({'entity_id':r['id'], 
                            'entity_type':r['value']['labels'][0], 
                            'entity_text':r['value']['text'].replace('\n', ' '),
                            'start':r['value']['start'], 
                            'end':r['value']['end']})
    
    return entity_list
        
  
  def _parse_relation(self, idx:int) -> List[Dict[str, str]]:
    rel_list = []
    ann = self.annotation[idx]
    for r in ann['annotations'][0]['result']:
      if r['type']=='relation':
        
        rel_list.append({'relation_id':f"{ann['data'][self.doc_id_var]}_{r['from_id']}_{r['to_id']}",
                         'entity_1_id':r['from_id'], 
                         'entity_2_id':r['to_id']})
       
    return rel_list
    
  
  def pop_IE(self):
    """
    This method iterate through annotation files and create IEs
    """
    loop = tqdm(range(len(self.annotation)), total=len(self.annotation), leave=True)
    for i in loop:
      doc_id = self._parse_doc_id(i)
      txt = self._parse_text(i)
      entity = self._parse_entity(i)
      entity_text = {e['entity_id']:e['entity_text'] for e in entity}
      entity_type = {e['entity_id']:e['entity_type'] for e in entity}
      # check if there're overlapped entities
      if self._check_overlap_entities(entity):
        warnings.warn(f'Overlapped entity detected! Action: "{self.overlap_entity_action}" will be taken.')
        if self.overlap_entity_action == 'random':
          entity = self._random_sample_overlap_entities(entity)
        if self.overlap_entity_action == 'priority':
          entity = self._prioritize_overlap_entities(entity)
        elif self.overlap_entity_action == 'keep':
          warnings.warn('Overlapped entities were kept. Further processing is suggested.')
      
      relation_raw = self._parse_relation(i)
      # fill in relation_type, entity_1_text, entity_2_text in relations
      relation = []
      for r in relation_raw:
        r['entity_1_text'] = entity_text[r['entity_1_id']]
        r['entity_2_text'] = entity_text[r['entity_2_id']]
        if (entity_type[r['entity_1_id']], entity_type[r['entity_2_id']]) in self.relation_map:
          r['relation_type'] = self.relation_map[(entity_type[r['entity_1_id']], entity_type[r['entity_2_id']])]
          relation.append(r)
        
      ie = Information_Extraction_Document(doc_id=doc_id, 
                                           text=txt, 
                                           entity_list=entity, 
                                           relation_list=relation)
      
      ie.save(os.path.join(self.IE_dir, f'{doc_id}.ie'))


class BRAT_IE_converter(IE_converter):
  def __init__(self, txt_dir:str, ann_dir:str, IE_dir:str, overlap_entity_action:str='random', 
               overlap_entity_rank:List[str]=None):
    """
    This class inputs a directory with annotation files, outputs IEs
    Parameters
    ----------
    txt_dir: str
      Directory of text files
    ann_dir : str
      Directory of annotation files
    IE_dir : str
      Directory of IE files 
    """
    super().__init__(IE_dir, overlap_entity_action, overlap_entity_rank)
    self.txt_dir = txt_dir
    self.ann_dir = ann_dir

  
  def _parse_text(self, txt_filename:str):
    with open(os.path.join(self.txt_dir, txt_filename), 'r') as f:
      text = f.read()
    return text
      
  def _parse_entity(self, ann_filename:str) -> List[Dict[str, str]]:
    entity_list = []
    with open(os.path.join(self.ann_dir, ann_filename), 'r') as f:
      lines = f.readlines()
    for line in lines:
      if line[0] == 'T':
        l = line.split()
        entity_id = l[0]
        entity_type = l[1]
        start = int(l[2])
        i = 3
        while True:
          if ';' in l[i]:
            i += 1
          else:
            end = int(l[i])
            break
            
        entity_list.append({'entity_id':entity_id, 
                            'entity_type':entity_type, 
                            'start':start, 
                            'end':end})
        
    return entity_list
  
  
  def _parse_relation(self, ann_filename:str) -> List[Dict[str, str]]:
    rel_list = []
    with open(os.path.join(self.ann_dir, ann_filename), 'r') as f:
      lines = f.readlines()
    for line in lines:
      if line[0] == 'R':
        res = {}
        vec = line.split()
        res['relation_id'] = vec[0]
        res['relation_type'] = vec[1]
        res['entity_1_id'] = vec[2].replace('Arg1:', '')
        res['entity_2_id'] = vec[3].replace('Arg2:', '')
        rel_list.append(res)
    return rel_list
    
  
  def pop_IE(self):
    """
    This method iterate through annotation files and create IEs
    """
    files = sorted([f.replace('.ann', '') for f in os.listdir(self.ann_dir) 
                 if os.path.isfile(os.path.join(self.ann_dir, f)) and f[-4:] == '.ann'])
    loop = tqdm(files, total=len(files), leave=True)
    for file in loop:
      txt = self._parse_text(f'{file}.txt')
      entity = self._parse_entity(f'{file}.ann')
      # check if there're overlapped entities
      if self._check_overlap_entities(entity):
        warnings.warn(f'Overlapped entity detected! Action: "{self.overlap_entity_action}" will be taken.')
        if self.overlap_entity_action == 'random':
          entity = self._random_sample_overlap_entities(entity)
        if self.overlap_entity_action == 'priority':
          entity = self._prioritize_overlap_entities(entity)
        elif self.overlap_entity_action == 'keep':
          warnings.warn('Overlapped entities were kept. Further processing is suggested.')
      # Remove newline in entity text  
      for e in entity:
        e['entity_text'] = txt[e['start']:e['end']].replace('\n', ' ')
      # Process relations
      entity_text_dict = {e['entity_id']:e['entity_text'] for e in entity}
      relation = self._parse_relation(f'{file}.ann')
      # check if both entity_1 and entity_2 are in entity list. If not, drop relation. 
      filter_relation = []
      for r in relation:
        if r['entity_1_id'] in entity_text_dict and r['entity_2_id'] in entity_text_dict:          
          r['entity_1_text'] = entity_text_dict[r['entity_1_id']]
          r['entity_2_text'] = entity_text_dict[r['entity_2_id']]
          filter_relation.append(r)
       
      ie = Information_Extraction_Document(doc_id=file, 
                                           text=txt, 
                                           entity_list=entity, 
                                           relation_list=filter_relation)
      ie.save(os.path.join(self.IE_dir, f'{file}.ie'))


class Trainer():
  def __init__(self, 
               run_name: str, 
               model: Union[AutoModelForTokenClassification, AutoModelForSequenceClassification], 
               n_epochs: int, 
               train_dataset: Dataset, 
               batch_size: int, 
               optimizer, 
               valid_dataset: Dataset=None, 
               shuffle: bool=True, 
               drop_last: bool=True,
               save_model_mode: str="best", 
               save_model_path: str=None, 
               early_stop: bool=True,
               early_stop_epochs: int=8,
               log_path: str=None, 
               device:str=None):    
    """
    This class trains a model with taining dataset and outputs checkpoints.

    Parameters
    ----------
    run_name : str
      Name of the run/ experiment.
    model : Union[AutoModelForTokenClassification, AutoModelForSequenceClassification]
      A base model to train.
    n_epochs : int
      Number of epochs.
    train_dataset : Dataset
      Training dataset.
    batch_size : int
      Batch size.
    optimizer : TYPE
      optimizer.
    valid_dataset : Dataset, optional
      Validation dataset. The default is None.
    shuffle : bool, optional
      Indicator for shuffling training instances. The default is True.
    drop_last : bool, optional
      Drop last training batch, so the shape of each batch is same. The default is True.
    save_model_mode : str, optional
      Must be one of {"best", "all"}. The default is best.
    save_model_path : str, optional
      Path for saving checkpoints. The default is None.
    early_stop : bool, optional
      Indicator for early stop
    early_stop_epochs : int, optional
      if early_stop=True, a continuous n epoches that the validation loss does not 
      drop will result in early stop.
    log_path : str, optional
      Path for saving logs. The default is None.
    device : str, optional
      CUDA device name. The default is cuda:0 if available, or cpu.
    """
    if device:
      self.device = device
    else: 
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
    self.run_name = run_name
    self.model = model
    self.model.to(self.device)
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.valid_dataset = valid_dataset
    self.shuffle = shuffle
    self.save_model_mode = save_model_mode
    self.save_model_path = os.path.join(save_model_path, self.run_name)
    if save_model_path != None and not os.path.isdir(self.save_model_path):
      os.makedirs(self.save_model_path)
    self.best_loss = float('inf')
    self.early_stop = early_stop
    self.loss_no_drop_epochs = 0
    self.early_stop_epochs = early_stop_epochs
    self.train_dataset = train_dataset
    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                   shuffle=self.shuffle, drop_last=drop_last)
    if valid_dataset != None:
      self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                     shuffle=False, drop_last=drop_last)
    else:
      self.valid_loader = None
    
    self.log_path = os.path.join(log_path, self.run_name)
    if log_path != None and not os.path.isdir(self.log_path):
      os.makedirs(self.log_path)
    self.tensorboard_writer = SummaryWriter(self.log_path) if log_path != None else None
    self.global_step = 0
    
  
  def save_model(self, epoch, train_loss, valid_loss):
    torch.save(self.model.state_dict(), 
               os.path.join(self.save_model_path, 
                            f'Epoch-{epoch}_trainloss-{train_loss:.4f}_validloss-{valid_loss:.4f}.pth'))
  
    
  @abc.abstractmethod
  def evaluate(self) -> float:
    """
    This method evaluates model with validation set

    Returns
    -------
    float
      average loss across validation set.
    """
    return NotImplemented
    
  @abc.abstractmethod  
  def train(self):
    """
    This method trains, (validate) model and saves checkpoints.
    """
    return NotImplemented
            

class IE_Evaluator:
  def __init__(self, pred_IEs:List[Information_Extraction_Document], 
               gold_IEs:List[Information_Extraction_Document]):
    """
    This class evaluates a List of predicted IE objects against corresponding gold IE objects
    Outputs ddocument-level and overall level evaluation reports.

    Parameters
    ----------
    pred_IEs : List[Information_Extraction_Document]
      a list of predicted IEs.
    gold_IEs : List[Information_Extraction_Document]
      a list of gold IEs that includes the predicted IE documents. 
      Can be more than predicted documents. 
    """
    assert set([e['doc_id'] for e in pred_IEs]).issubset(set([e['doc_id'] for e in gold_IEs])), \
    "Gold IEs must include all Predicted IEs"
    self.pred_IEs_dict = {ie['doc_id']:ie for ie in pred_IEs}
    self.gold_IEs_dict = {ie['doc_id']:ie for ie in gold_IEs}
    
    # Get all entity types and relation types in Gold standard
    self.entity_types = []
    self.relation_types = []
    for ie in gold_IEs:
      if ie.has_entity():
        for e in ie['entity']:
          if e['entity_type'] not in self.entity_types:
            self.entity_types.append(e['entity_type'])
          
      if ie.has_relation():
        for e in ie['relation']:
          if e['relation_type'] not in self.relation_types:
            self.relation_types.append(e['relation_type'])
      
  
  def F1(self, p:float, r:float):
    if p+r == 0:
      return float('nan')
    
    return 2*p*r/(p+r)

    
  def _NER_evaluate(self, pred_entities:List[Dict], 
                      gold_entities:List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    This method evaluates a single IE document against its gold standard.
    Outputs a dict of metrics.
  
    Parameters
    ----------
    pred_ie : List[Dict]
      List of entities to evaluate.
    gold_ie : List[Dict]
      List of entities for Gold standard.
  
    Returns
    -------
    a dict {entity_type: {gold, pred, exact, partial_in_gold, partial_in_pred, exact_precision,
    exact_recall, exact_F1, partial_precision, partial_recall, partial_F1}}
    the dict also has {'<Average>': {exact_micro_F1, partial_micro_F1, exact_unweighted_macro_F1, 
    patial_unweighted_macro_F1, exact_weighted_macro_F1, partial_weighted_macro_F1}}

    metrics for each entity types:
      gold: number of gold standard entities
      pred: number of predicted entities
      exact: number of predicted entities exactly match gold entities
      partial_in_gold: number of predicted entities partially overlap with gold entities
                        each gold entity will only be counted once
      partial_in_pred: number of predicted entities partially overlap with gold entities
                        each predicted entity will only be counted once
      exact_precision: precision in exact mode
      exact_recall: recall in exact mode
      exact_F1: F1 score in exact mode
      partial_precision: precision in partial mode
      partial_recall: recall in partial mode
      partial_F1: F1 score in partial mode
    Average F1
      exact_micro_F1: total (exact) correct entities/ total entities
      partial_micro_F1: total (partial) correct entities/ total entities
      exact_unweighted_macro_F1: raw average of f1 scores of all entity types
      partial_unweighted_macro_F1: raw average of f1 scores of all entity types
      exact_weighted_macro_F1: weighted average of f1 scores of all entity types
      partial_weighted_macro_F1: weighted average of f1 scores of all entity types
    """
    entity_type_res = {t:{'gold':0, 'pred':0, 'exact':0, 'partial_in_gold':0, 'partial_in_pred':0,
              'exact_precision':float('nan'), 'exact_recall':float('nan'), 'exact_F1':float('nan'),
              'partial_precision':float('nan'), 'partial_recall':float('nan'), 'partial_F1':float('nan')} 
           for t in self.entity_types}
    
    # Count predicted entities
    for pred_entity in pred_entities:
      entity_type_res[pred_entity['entity_type']]['pred'] += 1
      
    # Count gold entities
    for gold_entity in gold_entities:
      entity_type_res[gold_entity['entity_type']]['gold'] += 1
    
    # Count matches per pred
    for pred_entity in pred_entities:
      for gold_entity in gold_entities:
        if pred_entity['entity_type'] == gold_entity['entity_type']:
          # exact
          if (pred_entity['start'] == gold_entity['start']) and (pred_entity['end'] == gold_entity['end']):
            entity_type_res[gold_entity['entity_type']]['exact'] += 1
          # partial
          if not(pred_entity['end'] < gold_entity['start'] or pred_entity['start'] > gold_entity['end']):
            entity_type_res[gold_entity['entity_type']]['partial_in_pred'] += 1
            break
      
    # Count matches per gold
    for gold_entity in gold_entities:
      for pred_entity in pred_entities:
        if pred_entity['entity_type'] == gold_entity['entity_type']:
          # partial
          if not(pred_entity['end'] < gold_entity['start'] or pred_entity['start'] > gold_entity['end']):
            entity_type_res[gold_entity['entity_type']]['partial_in_gold'] += 1
            break
    
    # Calculate per-entity-type precision, recall, F1    
    for entity_type, d in entity_type_res.items():
      entity_type_res[entity_type]['exact_precision'] = d['exact']/ d['pred'] if (d['pred'] != 0) else float('nan')
      entity_type_res[entity_type]['partial_precision'] = d['partial_in_pred']/ d['pred'] if (d['pred'] != 0) else float('nan')
      entity_type_res[entity_type]['exact_recall'] = d['exact']/ d['gold'] if (d['gold'] != 0) else float('nan')
      entity_type_res[entity_type]['partial_recall'] = d['partial_in_gold']/ d['gold'] if (d['gold'] != 0) else float('nan')
      entity_type_res[entity_type]['exact_F1'] = self.F1(entity_type_res[entity_type]['exact_precision'], entity_type_res[entity_type]['exact_recall'])
      entity_type_res[entity_type]['partial_F1'] = self.F1(entity_type_res[entity_type]['partial_precision'], entity_type_res[entity_type]['partial_recall'])
  
    # Calculate average F1 (micro, unweighted macro, weighted macro)
    total_exact = 0
    total_partial = 0
    total_gold = 0
    total_exact_f1 = 0
    total_partial_f1 = 0
    total_weighted_exact_f1 = 0
    total_weighted_partial_f1 = 0
    total_gold_entities = sum([d['gold'] for entity_type, d in entity_type_res.items()])
    none_na_exact_f1 = 0
    none_na_partial_f1 = 0
    for entity_type, d in entity_type_res.items():
      total_exact += d['exact']
      total_partial += d['partial_in_gold']
      total_gold += d['gold']
      if pd.notna(entity_type_res[entity_type]['exact_F1']):
        none_na_exact_f1 += 1
        total_exact_f1 += entity_type_res[entity_type]['exact_F1']
        total_weighted_exact_f1 += entity_type_res[entity_type]['exact_F1'] * (d['gold']/total_gold_entities)
      if pd.notna(entity_type_res[entity_type]['partial_F1']):
        none_na_partial_f1 += 1
        total_partial_f1 += entity_type_res[entity_type]['partial_F1']
        total_weighted_partial_f1 += entity_type_res[entity_type]['partial_F1'] * (d['gold']/total_gold_entities)

    
    average_res = {}
    average_res['exact_micro_F1'] = total_exact/total_gold if total_gold != 0 else float('nan')
    average_res['partial_micro_F1'] = total_partial/total_gold if total_gold != 0 else float('nan')
    average_res['exact_unweighted_macro_F1'] = total_exact_f1/ none_na_exact_f1 if none_na_exact_f1 != 0 else float('nan')
    average_res['partial_unweighted_macro_F1'] = total_partial_f1/ none_na_partial_f1 if none_na_partial_f1 != 0 else float('nan')
    average_res['exact_weighted_macro_F1'] = total_weighted_exact_f1/ none_na_exact_f1 if none_na_exact_f1 != 0 else float('nan')
    average_res['partial_weighted_macro_F1'] = total_weighted_partial_f1/ none_na_partial_f1 if none_na_partial_f1 != 0 else float('nan')

    return entity_type_res, average_res

  def NER_evaluate(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    This method iterate through all predicted IEs and output 
    1. Overall evaluation
    2. per-document evaluation

    Returns
    -------
    a dataframe with [gold, pred, exact, partial_in_gold, partial_in_pred, exact_precision,
                      exact_recall, exact_F1, partial_precision, partial_recall, partial_F1]
      each row is an entity type
    a Dict of {doc_id, dataframe} for each document
    """
    doc_res = {}
    for _, ie in self.pred_IEs_dict.items():
      pred_entities = ie['entity']
      gold_entities = self.gold_IEs_dict[ie['doc_id']]['entity']
      entity_type_res, average_res = self._NER_evaluate(pred_entities=pred_entities, gold_entities=gold_entities)
      table = pd.DataFrame(entity_type_res).transpose().reset_index().rename(columns={'index':'entity_type'})
      avg = pd.DataFrame({'entity_type':['<Micro f1 score>', '<Macro (unweighted) f1 score>', '<Macro (weighted) f1 score>'],
                          'exact_F1': [average_res['exact_micro_F1'], average_res['exact_unweighted_macro_F1'], average_res['exact_weighted_macro_F1']],
                          'partial_F1': [average_res['partial_micro_F1'], average_res['partial_unweighted_macro_F1'], average_res['partial_weighted_macro_F1']]
                        })
      table = pd.concat([table, avg], axis=0)
      doc_res[ie['doc_id']] = table
      
    total = pd.concat([table for _, table in doc_res.items()])
    total_res = total.groupby('entity_type').agg({'gold':'sum', 'pred':'sum', 'exact':'sum',
                                              'partial_in_gold':'sum', 'partial_in_pred':'sum'})
    
    total_res.reset_index(inplace=True)
    total_res = total_res[~total_res['entity_type'].isin(['<Macro (unweighted) f1 score>', 
                                                          '<Macro (weighted) f1 score>',
                                                          '<Micro f1 score>'])]
    total_res['exact_precision'] = total_res['exact']/total_res['pred']
    total_res['partial_precision'] = total_res['partial_in_pred']/total_res['pred']
    total_res['exact_recall'] = total_res['exact']/total_res['gold']
    total_res['partial_recall'] = total_res['partial_in_gold']/total_res['gold']
    total_res['exact_F1'] = total_res.apply(lambda x:self.F1(x.exact_precision, x.exact_recall), axis=1)
    total_res['partial_F1'] = total_res.apply(lambda x:self.F1(x.partial_precision, x.partial_recall), axis=1)
    # Micro F1
    exact_micro_f1 = total_res['exact'].sum()/ total_res['gold'].sum()
    partial_micro_f1 = total_res['partial_in_gold'].sum()/ total_res['gold'].sum()
    # unweighted Macro
    exact_unweighted_macro_f1 = total_res['exact_F1'].mean()
    partial_unweighted_macro_f1 = total_res['partial_F1'].mean()
    # weighted Macro
    total_gold_entities = total_res['gold'].sum()
    exact_weighted_macro_f1 = (total_res['exact_F1'] * total_res['gold']/ total_gold_entities).sum()
    partial_weighted_macro_f1 = (total_res['partial_F1'] * total_res['gold']/ total_gold_entities).sum()
    
    avg = pd.DataFrame({'entity_type':['<Micro f1 score>', '<Macro (unweighted) f1 score>', '<Macro (weighted) f1 score>'], 
                       'exact_F1': [exact_micro_f1, exact_unweighted_macro_f1, exact_weighted_macro_f1], 
                       'partial_F1': [partial_micro_f1, partial_unweighted_macro_f1, partial_weighted_macro_f1]})

    total_res = pd.concat([total_res, avg], axis=0)

    return total_res, doc_res
  

  def RE_evaluate(self, pred_relations:List[Dict], 
                  gold_relations:List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    This method evaluates a single IE document against its gold standard.
    Outputs a dict {relation_type: {gold, pred, precision, recall, F1}}
  
    Parameters
    ----------
    pred_ie : List[Dict]
      List of entities to evaluate.
    gold_ie : List[Dict]
      List of entities for Gold standard.
  
    Returns
    -------
    a dict {relation_type: {gold, pred, precision, recall, F1}
    """
    res = {r:{'gold':0, 'pred':0, 'match':0, 'precision':float('nan'), 'recall':float('nan'), 'F1':float('nan')} 
           for r in self.relation_types}
    # Count predicted relations
    for pred_relation in pred_relations:
      res[pred_relation['relation_type']]['pred'] += 1
      
    # Count gold relations
    for gold_relation in gold_relations:
      res[gold_relation['relation_type']]['gold'] += 1
      
    # Count matches
    for gold_relation in gold_relations:
      for pred_relation in pred_relations:
        if pred_relation['relation_type'] == gold_relation['relation_type']:
          if sorted([pred_relation['entity_1_id'], pred_relation['entity_2_id']]) == \
              sorted([gold_relation['entity_1_id'], gold_relation['entity_2_id']]):
            res[gold_relation['relation_type']]['match'] += 1
            break
    
    # Calculate precision, recall, F1    
    for relation_type, d in res.items():
      res[relation_type]['precision'] = d['match']/ d['pred'] if (d['pred'] != 0) else float('nan')
      res[relation_type]['recall'] = d['match']/ d['gold'] if (d['gold'] != 0) else float('nan')
      res[relation_type]['F1'] = self.F1(res[relation_type]['precision'], res[relation_type]['recall'])
    
    return res
  
  
  def RE_evaluate_All(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    This method iterate through all predicted IEs and output 
    1. Overall evaluation
    2. per-document evaluation

    Returns
    -------
    a dataframe with columns [gold, pred, match, precision, recall, F1],
      each row is an relation type
    a Dict of {doc_id, dataframe} for each document
    """
    doc_res = {}
    for _, ie in self.pred_IEs_dict.items():
      pred_relations = ie['relation']
      gold_relations = self.gold_IEs_dict[ie['doc_id']]['relation']
      res = self.RE_evaluate(pred_relations=pred_relations, gold_relations=gold_relations)

      table = pd.DataFrame(res).transpose().reset_index().rename(columns={'index':'relation_type'})
      doc_res[ie['doc_id']] = table
      
    total = pd.concat([table for _, table in doc_res.items()])
    total_res = total.groupby('relation_type').agg({'gold':'sum', 'pred':'sum', 'match':'sum'})
    
    total_res.reset_index(inplace=True)
    total_res['precision'] = total_res['match']/total_res['pred']
    total_res['recall'] = total_res['match']/total_res['gold']
    total_res['F1'] = total_res.apply(lambda x:self.F1(x.precision, x.recall), axis=1)
    
    return total_res, doc_res
  