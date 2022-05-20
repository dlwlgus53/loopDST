from data_augment3.augment import seed_setting
import sys
# get from  data_augment3
# get from learn.py
import torch
import random
import numpy as np
import json
import ontology
import random
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class aug_training:
    def __init__(self,top_n, seed):
        log = log_setting()
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
            
            
    def _change_format(raw_data, labeled,data):
        return raw_data
    
    def augment(self, raw_data, labeled_data, init_labeled_data, change_rate): # 이미 있는걸 받았다고 쳐
        raw_data = _change_format(raw_data, labeled_data)
        
        # 이거 그대로 로드해오기
        dial_turn_id_list, tokenized_masked_list = get_will_change_item(raw_data,init_labeled_data, self.aug_tokenizer, change_rate)
        generated_dict= generate_new_text(model, dial_turn_id_list, tokenized_masked_list, args.batch_size, DEVICE)
        
        raw_data_similar = []
        for dial_idx, dial in enumerate(raw_data):
            dial_turn_key = '[d]'+ dial[0]['dial_id'] + '[t]0'
            if dial_turn_key not in init_labeled_data: continue
            if dial_idx%30 == 0 and dial_idx !=0:log.info(f'saving dials {dial_idx}/{len(raw_data)} done')
            for n in range(args.topn+1):
                if n==0:
                    similar_dial = copy.deepcopy(dial)
                else:
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
