# -*- coding: utf-8 -*-
import abc
from typing import List, Tuple, Dict, Optional
import pandas as pd
from modules.Utilities import Information_Extraction_Document, Trainer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from itertools import combinations


def binary_search(position:int, spans:Tuple[int, int]) -> int:
  """
  This function searches the span for a given position. 
  returns the index of the span

  Parameters
  ----------
  position : int
    The entity start position.
  spans : Tuple[int, int]
    a list of token spans (tuples).

  Returns
  -------
  mid : TYPE
    the span index.
  """
  left = 0
  right = len(spans) - 1
  while left <= right:
    mid = (left + right) // 2
    span_start, span_end = spans[mid]
    if span_start <= position <= span_end:
      return mid
    elif position < span_start:
      right = mid - 1
    else:
      left = mid + 1

  return None

class RE_Dataset(Dataset):
  def __init__(self, 
               IEs: List[Information_Extraction_Document], 
               tokenizer: AutoTokenizer, 
               possible_rel: List[List[str]],
               token_length: int,
               label_map: Dict,
               has_label: bool=True):
    """
    This parent class inputs list of IEs and for any combination of 2 entities, 
    outputs the segment of tokens (tokenizer) as dict 
    {document_id, input_ids, attention_mask, (labels)}
    number of tokens per input is padded/ truncated to token_length

    Parameters
    ----------
    IEs : List[Information_Extraction_Document]
      Dict of IE.
    tokenizer : AutoTokenizer
      tokenizer.
    possible_rel : List[List[str]]
      List of 2-List of entity types to indicate possible relations. Only combination 
      in this list will be modeled. Others are assumed No_relation
    token_length : int
      The max token distance between 2 entities to consider relations, this is 
      also the max token length for each input.
    label_map : Dict
      Dict of {entity type: id}
    has_label : bool, optional
      Indicates if the IE has entity. The default is True.
    """
    self.IEs = IEs
    self.tokenizer = tokenizer
    self.token_length = token_length
    self.possible_rel = set([tuple(rel)for rel in possible_rel])
    self.label_map = label_map
    self.has_label = has_label
    
    self.token_spans = {}
    self.entity_spans = {}
    self._get_spans()
    
    self.segments = []
    self.get_segments()
    
  
  @abc.abstractmethod
  def _get_segments(self, ie:Information_Extraction_Document) -> List[Dict[str, str]]:
    """
    This method inputs an IE and outputs a list of segments, each correspond to 
    a pair of entities.

    Parameters
    ----------
    ie : Information_Extraction_Document
      IE for the document.

    Returns
    -------
    List[Dict[str, str]]
      a list of segments as Dict {doc_id, segment, entity_1_id, entity_2_id, (relation_type)}.
    """
    return NotImplemented
    
  
  def _get_spans(self):
    loop = tqdm(self.IEs, total=len(self.IEs), leave=True)
    loop.set_description('Calculate token spans')
    for ie in loop:
      self.token_spans[ie['doc_id']] = None
      self.entity_spans[ie['doc_id']] = {}
      tokens = self.tokenizer(ie['text'], add_special_tokens=True, return_offsets_mapping=True) 
      self.token_spans[ie['doc_id']] = tokens.offset_mapping
      for e in ie['entity']:
        self.entity_spans[ie['doc_id']][e['entity_id']] = binary_search(e['start'], tokens.offset_mapping)
      
      
  def get_segments(self):
    loop = tqdm(self.IEs, total=len(self.IEs), leave=True)
    loop.set_description('Prepare segments')
    for ie in loop:
      self.segments.extend(self._get_segments(ie))
  
  
  def _get_relation_type(self, relations:List[Dict], entity_1_id:str, entity_2_id:str) -> str:
    """
    This method gets the relation type between 2 entities
    """
    for r in relations:
      if sorted([entity_1_id, entity_2_id]) == sorted([r['entity_1_id'], r['entity_2_id']]):
        return r['relation_type']
    
    return 'No_relation'
    
  
  def _check_N_tokens_between(self, doc_id, entity_1_id, entity_2_id) -> bool:
    """
    This method calculates the number of tokens between 2 entities, 
    including the 2 entities, and return if < token_length
    """
    return abs(self.entity_spans[doc_id][entity_1_id] - \
               self.entity_spans[doc_id][entity_2_id]) < self.token_length
  
  
  def _check_possible_rel(self, entity_1_type:str, entity_2_type:str) -> bool:
    """
    This method checks if the 2 entities are possible to have relation
    """
    return (entity_1_type, entity_2_type) in self.possible_rel or \
            (entity_2_type, entity_1_type) in self.possible_rel
  
  
  def __len__(self):
    return len(self.segments)
  
  
  def __getitem__(self, idx:int) -> Dict:
    # get segment
    seg = self.segments[idx]
    # tokenize segment
    tokens = self.tokenizer(seg['segment'], 
                            padding='max_length',
                            max_length=self.token_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_token_type_ids=False)
    
    tokens['input_ids'] = torch.tensor(tokens['input_ids'])
    tokens['attention_mask'] = torch.tensor(tokens['attention_mask'])
    tokens['doc_id'] = seg['doc_id']
    tokens['entity_1_id'] = seg['entity_1_id']
    tokens['entity_2_id'] = seg['entity_2_id']
    
    if self.has_label:
      tokens['label'] = torch.tensor(self.label_map[seg['relation_type']])
    
    return tokens


class InlineTag_RE_Dataset(RE_Dataset):
  def __init__(self, 
               IEs: List[Information_Extraction_Document], 
               tokenizer: AutoTokenizer, 
               possible_rel: List[List[str]],
               token_length: int,
               label_map: Dict,
               has_label: bool=True):
    """
    This class impliments a published method DOI: 10.18653/v1/W19-1908
    
    Special tokens are placed inline the text to indicate begin/ end of entities
    below used "es" for event start, "ee" for event end, "ts" for time expression
    start, "te" for time expression end.
    
    #1: . a es surgery ee was scheduled on ts date te .
    #2: . a surgery was es scheduled ee on ts date te .
    #3: . a eas surgery eae was ebs scheduled ebe on march

    We use "[E]" and "[\E]" to indicate entity start and end
    """    
    super().__init__(IEs, tokenizer, possible_rel, token_length, label_map, has_label)
    
  def _get_segments(self, ie:Information_Extraction_Document) -> List[Dict[str, str]]:
    segments = []
    for entity_1, entity_2 in combinations(ie['entity'], 2):
      within_N_tokens = self._check_N_tokens_between(ie['doc_id'], entity_1['entity_id'], entity_2['entity_id'])
      within_possible_rel = self._check_possible_rel(entity_1['entity_type'], entity_2['entity_type'])
      
      if within_N_tokens and within_possible_rel:
        mid_token_pos = (self.entity_spans[ie['doc_id']][entity_1['entity_id']] + \
                         self.entity_spans[ie['doc_id']][entity_2['entity_id']]) //2
        
        start_token_pos = max(0, mid_token_pos - self.token_length//2) + 1
        end_token_pos = min(mid_token_pos + self.token_length//2, len(self.token_spans[ie['doc_id']]) - 1)
        
        start_pos = self.token_spans[ie['doc_id']][start_token_pos][0]
        end_pos = self.token_spans[ie['doc_id']][end_token_pos][1]
        first_entity, second_entity = sorted([entity_1, entity_2], key=lambda x:x['start'])
        
        segment = ie['text'][start_pos:first_entity['start']] + " [E] " + \
                  ie['text'][first_entity['start']:first_entity['end']] + " [\E] " + \
                  ie['text'][first_entity['end']:second_entity['start']] + " [E] " + \
                  ie['text'][second_entity['start']:second_entity['end']] + " [\E] " + \
                  ie['text'][second_entity['end']:end_pos]
        
        if self.has_label:
          segments.append({'doc_id':ie['doc_id'], 'segment':segment, 
                           'entity_1_id':entity_1['entity_id'], 'entity_2_id':entity_2['entity_id'],
                           'relation_type':self._get_relation_type(ie['relation'], 
                                                                   entity_1['entity_id'], entity_2['entity_id'])})
        else:
          segments.append({'doc_id':ie['doc_id'], 'segment':segment, 
                           'entity_1_id':entity_1['entity_id'], 'entity_2_id':entity_2['entity_id']})
    
    return segments
    
    
class RE_Trainer(Trainer):
  def __init__(self, 
               run_name: str, 
               model: AutoModelForSequenceClassification, 
               n_epochs: int, 
               train_dataset: Dataset, 
               batch_size: int, 
               optimizer, 
               valid_dataset: Dataset=None, 
               shuffle: bool=True, 
               drop_last: bool=True,
               save_model_mode: str="best", 
               save_model_path: str=None, 
               log_path: str=None, 
               device:str=None):    
    """
    This class trains an RE model with taining dataset and outputs checkpoints.
    """
    super().__init__(run_name, model, n_epochs, train_dataset, batch_size, optimizer,
                     valid_dataset, shuffle, drop_last, save_model_mode, save_model_path,
                     log_path, device)
    
    
  def evaluate(self):
    with torch.no_grad():
      valid_total_loss = 0
      for valid_batch in self.valid_loader:
        valid_input_ids = valid_batch['input_ids'].to(self.device)
        valid_attention_mask = valid_batch['attention_mask'].to(self.device)
        valid_labels = valid_batch['label'].to(self.device)
        output = self.model(input_ids=valid_input_ids, 
                            attention_mask=valid_attention_mask, 
                            labels=valid_labels)
        valid_loss = output.loss
        valid_total_loss += valid_loss.item()
      return valid_total_loss/ len(self.valid_loader)
    
  def train(self):
    for epoch in range(self.n_epochs):
      train_total_loss = 0
      valid_mean_loss = None
      loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
      
      for batch_id, train_batch in loop:
        self.optimizer.zero_grad()
        self.global_step += 1
        train_input_ids = train_batch['input_ids'].to(self.device)
        train_attention_mask = train_batch['attention_mask'].to(self.device)
        train_labels = train_batch['label'].to(self.device)
        """ forward """
        output = self.model(input_ids=train_input_ids, 
                            attention_mask=train_attention_mask, 
                            labels =train_labels)
        train_loss = output.loss
        train_total_loss += train_loss.item()
        """ record training log """
        if self.tensorboard_writer != None:
          self.tensorboard_writer.add_scalar("train/loss", train_total_loss/ (batch_id+1), self.global_step)
        """ backward """
        train_loss.backward()
        """ update """
        self.optimizer.step()
        
        """ validation loss at end of epoch"""
        if self.valid_loader != None and batch_id == len(self.train_loader) - 1:
          valid_mean_loss = self.evaluate()
          if self.tensorboard_writer != None:
            self.tensorboard_writer.add_scalar("valid/loss", valid_mean_loss, self.global_step)
        """ print """
        train_mean_loss = train_total_loss / (batch_id+1)
        loop.set_description(f'Epoch [{epoch + 1}/{self.n_epochs}]')
        loop.set_postfix(train_loss=train_mean_loss, valid_loss=valid_mean_loss)
        
      """ end of epoch """
      if self.save_model_mode == 'all':
        self.save_model(epoch, train_mean_loss, valid_mean_loss)
      elif self.save_model_mode == 'best':
        if epoch == 0 or valid_mean_loss < self.best_loss:
          self.save_model(epoch, train_mean_loss, valid_mean_loss)
          
      self.best_loss = min(self.best_loss, valid_mean_loss)
            

class RE_Predictor:
  def __init__(self, 
               model:AutoModelForSequenceClassification,
               tokenizer:AutoTokenizer, 
               dataset: Dataset,
               label_map:Dict,
               batch_size:int,
               device:str=None):
    """
    This class inputs a fine-tuned model and a dataset. 
    outputs a list of IEs with entities (same as input), relations and probability
    {relation_id, relation_type, relation_prob, entity_1_id, entity_2_id, entity_1_text, entity_2_text}

    Parameters
    ----------
    model : AutoModelForSequenceClassification
      A model to make prediction.
    tokenizer : AutoTokenizer
      tokenizer.
    dataset : Dataset
      an (unlabeled) dataset for prediction.
    label_map : Dict
      DESCRIPTION.
    batch_size : int
      batch size for prediction. Does not affect prediction results.
    device : str, optional
      CUDA device name. The default is cuda:0 if available, or cpu.
    """
    if device:
      self.device = device
    else: 
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
    self.model = model
    self.model.to(self.device)
    self.model.eval()
    self.tokenizer = tokenizer
    self.label_map = label_map
    self.batch_size = batch_size
    self.dataset = dataset
    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

  def predict(self) -> List[Information_Extraction_Document]:
    """
    This method outputs a dict of IEs {doc_id, IE} with relations
    """
    prob_list = []
    entity_pair_list = []
    loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=True)
    for i, batch in loop:  
      input_ids = batch['input_ids'].to(self.device)
      attention_mask = batch['attention_mask'].to(self.device)
      with torch.no_grad():
        p = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pred = p.logits.softmax(dim=-1).cpu().tolist()
        entity_pair_list.extend(list(zip(batch['doc_id'], batch['entity_1_id'], batch['entity_2_id'])))
        prob_list.extend(pred)
    
    
    pair_df = pd.DataFrame(entity_pair_list, columns=['doc_id', 'entity_1_id', 'entity_2_id'])
    prob_df = pd.DataFrame(prob_list, columns=self.label_map.keys())
    pair_df['pred'] = prob_df.idxmax(axis=1)
    pair_df['prob'] = prob_df.max(axis=1)
    # Remove No_relation entity pairs
    pair_df = pair_df.loc[pair_df['pred'] != 'No_relation'].reset_index(drop=True)
    # pair_df has columns {'doc_id', 'entity_1_id', 'entity_2_id', 'pred', 'prob'}
    return self._pairs_to_IEs(pair_df)
  
  
  def _pairs_to_IEs(self, pairs:pd.DataFrame) -> List[Information_Extraction_Document]:
    ies = {ie['doc_id']: Information_Extraction_Document(doc_id=ie['doc_id'], 
                                                         text=ie['text'], 
                                                         entity_list=ie['entity']) 
           for ie in self.dataset.IEs}
    
    for r in pairs.itertuples():
      relation_id = f'{r.doc_id}_{r.entity_1_id}_{r.entity_2_id}'
      entity_1_text = ies[r.doc_id].get_entity_by_id(r.entity_1_id)['entity_text']
      entity_2_text = ies[r.doc_id].get_entity_by_id(r.entity_2_id)['entity_text']
      
      ies[r.doc_id].relation.append({'relation_id':relation_id, 
                                     'relation_type':r.pred, 
                                     'relation_prob':r.prob,
                                     'entity_1_id':r.entity_1_id,
                                     'entity_2_id':r.entity_2_id,
                                     'entity_1_text':entity_1_text,
                                     'entity_2_text':entity_2_text})
    return list(ies.values())
  