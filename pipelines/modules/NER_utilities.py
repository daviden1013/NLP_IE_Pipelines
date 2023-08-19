# -*- coding: utf-8 -*-
import abc
from typing import List, Tuple, Dict
import pandas as pd
from modules.Utilities import Information_Extraction_Document, Trainer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification
from tqdm import tqdm
import spacy
import re


class NER_Dataset(Dataset):
  def __init__(self, 
               IEs: Dict[str, Information_Extraction_Document], 
               tokenizer: AutoTokenizer, 
               token_length: int,
               label_map: Dict,
               has_label: bool=True,
               mode: str='BIO'):
    """
    This parent class inputs list of IEs and outputs dict 
    {document_id, input_ids, attention_mask, start, end, (labels)}
    number of tokens per input is padded/ truncated to token_length

    Parameters
    ----------
    IEs : Dict[str, Information_Extraction_Document]
      Dict of IEs with document_id as key, IE as value
    tokenizer : AutoTokenizer
      PretrainedTokenizerFast for the model
    token_length : int
      Number of tokens per input. 
    label_map : Dict
      Dict of {entity type: id}
    has_label : bool, optional
      Indicates if the IE has entity. The default is True.
    mode : str, optional
      The BIO mode, must be {"BIO", "IO"}. If has_label=False, mode will be ignored.
    """
    self.IEs = IEs
    self.tokenizer = tokenizer
    self.token_length = token_length
    self.label_map = label_map
    self.has_label = has_label
    assert mode in {"BIO", "IO"}, 'BIO mode must be one of {"BIO", "IO"}.'
    self.mode = mode
    self.segments = []
    self.get_segments()
    
  @abc.abstractmethod
  def _get_segments(self, doc_id:str, text:str) -> List[Dict[str, str]]:
    """
    This method segments a document text
    outputs a list of dict {doc_id, segment, start, end}
    """
    return NotImplemented
  
  def get_segments(self):
    loop = tqdm(self.IEs.items(), total=len(self.IEs.items()), leave=True)
    for _, ie in loop:
      self.segments.extend(self._get_segments(ie['doc_id'], ie['text']))
    
  def __len__(self) -> int:
    """
    This method outputs the total number of instances (segments)
    """
    return len(self.segments)
  
  def __getitem__(self, idx) -> Dict:
    """
    This method outputs a dict {doc_id, input_ids, attention_mask, start, end, (labels)}
    """
    # get segment
    seg = self.segments[idx]
    # tokenize segment
    tokens = self.tokenizer(seg['segment'], 
                            padding='max_length',
                            max_length=self.token_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_offsets_mapping=True,
                            return_token_type_ids=False)
    
    tokens['input_ids'] = torch.tensor(tokens['input_ids'])
    tokens['attention_mask'] = torch.tensor(tokens['attention_mask'])
    # calculate each tokens' span in the original document
    doc_span = []
    for offset in tokens['offset_mapping']:
      if offset == (0, 0):
        doc_span.append((0, 0))
      else:
        doc_span.append((offset[0] + seg['start'], offset[1] + seg['start']))
      
    tokens['spans'] = doc_span
    # Assign document ID
    tokens['doc_id'] = [seg['doc_id']] * self.token_length
    # Assign entity type label to each token
    if self.has_label:
      # only consider entities in this segment
      entity_list = [entity for entity in self.IEs[seg['doc_id']]['entity'] \
                     if entity['start'] >= seg['start'] and entity['end'] <= seg['end']]

      if self.mode == 'BIO':
        tokens['labels'] = self._get_BIO_entity_type(tokens['spans'], entity_list)
      elif self.mode == 'IO':
        tokens['labels'] = self._get_IO_entity_type(tokens['spans'], entity_list)
      
      tokens['labels'] = torch.tensor(tokens['labels'])
      
    return tokens
    
  
  def _is_in_entity(self, span:Tuple[int, int], start:int, end:int, criterion:str='contain') -> bool:
    """ 
    This method inputs a token span and a entity start, end 
    outputs whether the span is considered in entity
    Currently we have "overlap" or "contain" criteria
    """
    if criterion == 'overlap':
      return not(span[1] < start or span[0] > end)
    elif criterion == 'contain':
      return (start <= span[0] and end >= span[1])
    
  
  def _get_BIO_entity_type(self, spans:List[Tuple[int, int]], entity_list: List[Dict]) -> List[int]:
    """
    This method inputs a list of tokens span and entity list for the document
    outputs the entity type of that token in BIO mode

    Parameters
    ----------
    span : Tuple[int, int]
      token span
    entity_list : List[Dict]
      list of entities as dict {entity_id, entity_type, start, end}

    Returns
    -------
    str
      entity type id
    """
    span_labels = {}
    for entity in entity_list:
      is_first = True
      for span in spans:
        if self._is_in_entity(span, entity['start'], entity['end']):
          if is_first:
            pre_fix = 'B-'
            is_first = False
          else:
            pre_fix = 'I-'
          span_labels[span] = self.label_map[pre_fix + entity['entity_type']]
    
    return [0 if span not in span_labels else span_labels[span] for span in spans]
    
  
  def _get_IO_entity_type(self, spans:List[Tuple[int, int]], entity_list: List[Dict]) -> str:
    """
    This method inputs a list of tokens span and entity list for the document
    outputs the entity type of that token in IO mode

    Parameters
    ----------
    span : Tuple[int, int]
      token span
    entity_list : List[Dict]
      list of entities as dict {entity_id, entity_type, start, end}

    Returns
    -------
    str
      entity type name without pre-fix
    """
    span_labels = {}
    for entity in entity_list:
      for span in spans:
        if self._is_in_entity(span, entity['start'], entity['end']):
          span_labels[span] = self.label_map[entity['entity_type']]
    
    return [0 if span not in span_labels else span_labels[span] for span in spans]


class Sentence_NER_Dataset(NER_Dataset):
  def __init__(self, 
               IEs: Dict[str, Information_Extraction_Document], 
               tokenizer: AutoTokenizer, 
               token_length: int,
               label_map: Dict,
               has_label: bool=True,
               mode: str='BIO', 
               N_sentences:int=1):
    """
    This class 

    Parameters
    ----------
    IEs : Dict[str, Information_Extraction_Document]
      Dict of IEs with document_id as key, IE as value
    tokenizer : AutoTokenizer
      PretrainedTokenizerFast for the model
    token_length : int
      Number of tokens per input. 
    label_map : Dict
      Dict of {entity type: id}
    has_label : bool, optional
      Indicates if the IE has entity. The default is True.
    mode : str, optional
      The BIO mode, must be {"BIO", "IO"}. If has_label=False, mode will be ignored.
    N_sentences : int, optional
      Number of sentences to put together as an input. The default is 1.
    """
    
    self.nlp = spacy.load("en_core_web_sm")
    self.N_sentences = N_sentences
    # Remove unused pipelines to speed up
    for pipe in self.nlp.pipeline: 
      pipe_name = pipe[0]
      self.nlp.remove_pipe(pipe_name)
  
    self.sentencizer = self.nlp.add_pipe("sentencizer")
    
    super().__init__(IEs, tokenizer, token_length, label_map, has_label, mode)
    
  def _get_segments(self, doc_id:str, text:str) -> List[Dict[str, str]]:
    doc = self.nlp(text)
    sent_list = [s for s in doc.sents]
    sentences = []
    
    if self.N_sentences == 1:
      for sent in sent_list:
        sentences.append({'doc_id':doc_id, 'segment': sent.text.replace('\n', ' '), 
                          'start': sent.start_char, 'end': sent.end_char})
      
    else:  
      start = 0
      for i, sent in enumerate(sent_list):
        # The correct number of sentenes to output
        if (i + 1) % self.N_sentences == 0:        
          sentences.append({'doc_id':doc_id, 'segment': text[start:sent.end_char].replace('\n', ' '), 
                            'start': start, 'end': sent.end_char})   
        # The first sentence in the bundle
        elif i % self.N_sentences == 0:
          start = sent.start_char 
          
      # append the remaining sentences
      if len(sent_list) % self.N_sentences != 0:
        sentences.append({'doc_id':doc_id, 'segment': text[start:sent.end_char].replace('\n', ' '), 
                          'start': start, 'end': sent.end_char})  
        
    return sentences


class NER_Trainer(Trainer):
  def __init__(self, 
               run_name: str, 
               model: AutoModelForTokenClassification, 
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
    This class trains an NER model with taining dataset and outputs checkpoints.
    """
    super().__init__(run_name, model, n_epochs, train_dataset, batch_size, optimizer,
                     valid_dataset, shuffle, drop_last, save_model_mode, save_model_path,
                     log_path, device)
    

  def evaluate(self) -> float:
    with torch.no_grad():
      valid_total_loss = 0
      for valid_batch in self.valid_loader:
        valid_input_ids = valid_batch['input_ids'].to(self.device)
        valid_attention_mask = valid_batch['attention_mask'].to(self.device)
        valid_labels = valid_batch['labels'].to(self.device)
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
        train_labels = train_batch['labels'].to(self.device)
        # forward
        output = self.model(input_ids=train_input_ids, 
                            attention_mask=train_attention_mask, 
                            labels =train_labels)
        train_loss = output.loss
        train_total_loss += train_loss.item()
        # record training log
        if self.tensorboard_writer != None:
          self.tensorboard_writer.add_scalar("train/loss", train_total_loss/ (batch_id+1), self.global_step)
        # backward
        train_loss.backward()
        # update parameters
        self.optimizer.step()
        # validation loss at end of epoch
        if self.valid_loader != None and batch_id == len(self.train_loader) - 1:
          valid_mean_loss = self.evaluate()
          if self.tensorboard_writer != None:
            self.tensorboard_writer.add_scalar("valid/loss", valid_mean_loss, self.global_step)
        # print to progress bar
        train_mean_loss = train_total_loss / (batch_id+1)
        loop.set_description(f'Epoch [{epoch + 1}/{self.n_epochs}]')
        loop.set_postfix(train_loss=train_mean_loss, valid_loss=valid_mean_loss)
        
      # end of the epoch 
      if self.save_model_mode == 'all':
        self.save_model(epoch, train_mean_loss, valid_mean_loss)
      elif self.save_model_mode == 'best':
        if epoch == 0 or valid_mean_loss < self.best_loss:
          self.save_model(epoch, train_mean_loss, valid_mean_loss)
          
      self.best_loss = min(self.best_loss, valid_mean_loss)
            

class NER_Predictor:
  def __init__(self, 
               model:AutoModelForTokenClassification,
               tokenizer:AutoTokenizer, 
               dataset: Dataset,
               label_map:Dict,
               batch_size:int,
               device:str=None):
    """
    This class takes a model and an unlabeled dataset 
    outputs a dict of IEs {doc_id, IE} with entities

    Parameters
    ----------
    model : AutoModelForTokenClassification
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

  def predict(self) -> Dict[str, Information_Extraction_Document]:
    """
    This method outputs a dict of IEs {doc_id, IE} with entities
    """
    token_pred = {'doc_id':[],
                  'input_id':[],
                 'start':[],
                 'end':[]}
    for tag in self.label_map.keys():
      token_pred[f'prob_{tag}'] = []
    
    loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=True)
    for i, ins in loop:  
      input_ids = ins['input_ids'].to(self.device)
      attention_mask = ins['attention_mask'].to(self.device)
      with torch.no_grad():
        p = self.model(input_ids=input_ids, attention_mask=attention_mask)
        prob = p.logits.softmax(dim=-1)
        for b in range(prob.shape[0]):
          token_pred['doc_id'].extend([i[b] for i in ins['doc_id']])
          token_pred['input_id'].extend(ins['input_ids'][b].cpu().tolist())
          token_pred['start'].extend([i[0][b].item() for i in ins['spans']])
          token_pred['end'].extend([i[1][b].item() for i in ins['spans']])
          for tag, code in self.label_map.items():
            token_pred[f'prob_{tag}'].extend(prob[b,:,code].cpu().tolist())
    
    # df of predicted entities
    token_pred_df = pd.DataFrame(token_pred)
    # exlucde special tokens
    token_pred_df = token_pred_df.loc[~token_pred_df['input_id'].isin(self.tokenizer.all_special_ids)].\
      reset_index(drop=True)
    # Calculate predicted entity type
    token_pred_df['entity_type'] = token_pred_df[[f'prob_{v}' for v in self.label_map.keys()]].\
    idxmax(axis=1).str.replace('prob_', '')
    token_pred_df['prob'] = token_pred_df[[v for v in token_pred_df.columns if \
                                                       re.match('^prob_', v)]].max(axis=1)
      
    token_pred_df = token_pred_df[['doc_id', 'start', 'end', 'entity_type', 'prob']]
    
    entities = self._tokens_to_entities(token_df=token_pred_df)
    return self._entities_to_IEs(entities)


  def _tokens_to_entities(self, token_df: pd.DataFrame) -> pd.DataFrame:
    """
    This method inputs a df of tokens. must has columns: [doc_id, start, end, entity_type, prob]
    outputs a df of [doc_id, start, end, entity_type, prob, conf]
    """
    
    if self.dataset.mode == 'BIO':
      entity_chunk = (token_df['entity_type'].str.match('^B-|^O')).cumsum()
    else:
      entity_chunk = (token_df['entity_type'] != token_df['entity_type'].shift()).cumsum()
      
    entity = token_df.groupby(['doc_id', entity_chunk], sort=False, as_index=False).\
      agg({'start':'min', 'end':'max', 'entity_type':'max', 'prob':'mean'})
  
    entity = entity.loc[entity['entity_type'] != 'O']
    entity['entity_type'] = entity['entity_type'].str.replace('B-', '').str.replace('I-', '')
    # confident = prob / baseline prob
    entity['conf'] = entity['prob'] * len(self.label_map)
    return entity.reindex(['doc_id', 'start', 'end', 'entity_type', 'prob', 'conf'], axis=1)
    
  
  def _entities_to_IEs(self, entities: pd.DataFrame) -> Dict[str, Information_Extraction_Document]:
    ies = {doc_id: Information_Extraction_Document(doc_id=doc_id, text=ie['text']) 
           for doc_id, ie in self.dataset.IEs.items()}
    for r in entities.itertuples():
      entity_id = f'{r.doc_id}_{r.start}_{r.end}'
      entity_text = ies[r.doc_id]['text'][r.start:r.end].replace('\n', ' ')
      ies[r.doc_id].entity.append({'entity_id':entity_id, 
                                   'entity_text':entity_text, 
                                   'entity_type':r.entity_type,
                                   'start':r.start,
                                   'end':r.end})
    
    return ies
  