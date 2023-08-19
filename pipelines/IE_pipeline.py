# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import yaml
from modules.Utilities import BRAT_IE_converter
import logging
import pprint
  
def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  logging.info('IE pipeline started')
  
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
    
  logging.info('Config loaded:')
  pprint.pprint(config, sort_dicts=False)
  
  logging.info('Converting...')
  converter = BRAT_IE_converter(txt_dir=config['txt_dir'],
                                ann_dir=config['ann_dir'],
                                IE_dir=config['IE_dir'])
  
  converter.pop_IE()
  logging.info('IE pipeline finished')

if __name__ == '__main__':
  main()