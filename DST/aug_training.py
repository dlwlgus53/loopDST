import sys
sys.path.append('../')
from ..data_augment3.augment import *
import torch
import random
import copy
import numpy as np
import json
import ontology
import random
import os
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class Aug_training:
    def __init__(self,top_n, seed):
        seed_setting(seed)
        self.log = log_setting()
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
        self.aug_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.aug_config = RobertaConfig()
        self.aug_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(DEVICE)
    
    def _makedirs(self, path): 
        try: 
            os.makedirs(path) 
        except OSError: 
            if not os.path.isdir(path): 
                raise
            
            
    def _change_label(self, raw_data, labeled_data):
        for dial in raw_data:
            dial_key_idx = ""
            raw_data['bspn'] = labeled_data[dial_key_idx]
        return raw_data
    
    
    
    def augment(self, raw_data, labeled_data, change_rate, DEVICE, aug_batch_size = 10):
        raw_data = filtering_data(raw_data, labeled_data)
        raw_data = self._change_label(raw_data, labeled_data)
        dial_turn_id_list, tokenized_masked_list = get_will_change_item(raw_data, self.aug_tokenizer, change_rate)
        generated_dict= generate_new_text(self.aug_model, dial_turn_id_list, tokenized_masked_list, aug_batch_size, DEVICE)
        raw_data_similar = []
        
        for dial_idx, dial in enumerate(raw_data):
            if dial_idx%30 == 0 and dial_idx !=0: self.log.info(f'saving dials {dial_idx}/{len(raw_data)} done')
            for n in range(args.topn):
                similar_dial = []
                for turn in dial:
                    idx = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n-1)
                    similar_turn = copy.deepcopy(turn)
                    similar_turn['dial_id'] += f'_v{str(n)}'
                    similar_turn['user'] = generated_dict[idx]['text']
                    similar_turn['mask'] = generated_dict[idx]['mask_text']
                    similar_dial.append(similar_turn)
                raw_data_similar.append(similar_dial)
        return raw_data_similar
    
    def train(self, data, model, epoch):
        
        return model
