import sys
sys.path.append('../../../../')
from data_augment3.augment import log_setting, get_will_change_item, generate_new_text
import torch
import random
import copy
import numpy as np
import json
import ontology
import random
import os
import pdb
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from dataclass_part import DSTMultiWozData
from trainer_part import train, evaluate



all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class Aug_training:
    def __init__(self,aug_num, change_rate, data, device, log):
        log_setting("aug_log")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
        self.aug_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.aug_config = RobertaConfig()
        self.aug_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(DEVICE)
        self.aug_num = aug_num
        self.change_rate = change_rate
        self.log = log
        self.data = data
        self.device = device
    
    def _makedirs(self, path): 
        try: 
            os.makedirs(path) 
        except OSError: 
            if not os.path.isdir(path): 
                raise

    def augment(self):
        raw_data = self.data.replace_label(self.data.train_raw_data, self.data.labeled_data)
        id_list, masked_list = get_will_change_item(raw_data, self.aug_tokenizer, self.change_rate, self.aug_num, self.log)
        generated_dict= generate_new_text(self.aug_model, self.aug_tokenizer, id_list, masked_list, self.aug_num, self.device, self.log)
        raw_data_similar = []
        for dial_idx, dial in enumerate(raw_data):
            if dial_idx%30 == 0 and dial_idx !=0: self.log.info(f'saving dials {dial_idx}/{len(raw_data)} done')
            for n in range(self.aug_num):
                similar_dial = []
                for turn in dial:
                    idx = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                    similar_turn = copy.deepcopy(turn)
                    similar_turn['dial_id'] += f'_v{str(n)}'
                    if idx in generated_dict:
                        similar_turn['user'] = generated_dict[idx]['text']
                        similar_turn['mask'] = generated_dict[idx]['mask_text']
                    else:
                        pass
                    similar_dial.append(similar_turn)
                raw_data_similar.append(similar_dial)
        train = raw_data_similar[:int(len(raw_data_similar) * 0.9)]
        dev = raw_data_similar[int(len(raw_data_similar) * 0.9):]
        return train, dev
    
    def train(self, args, train_data, dev_data, model, epoch, optimizer, scheduler):
        device = torch.device(self.device)
        best_result = 0
        best_model = None
        self.data.set_train_aug(train_data)
        self.data.set_eval_aug(dev_data)

        for epoch in range(epoch):
            train_loss = train(args,model,optimizer, scheduler, self.data,self.log, cuda_available = True,  mode = 'train_aug')
            _, dev_score = evaluate(args,model,self.data,self.log, cuda_available = True, device = device,  mode = 'eval_aug')
            if dev_score > best_result:
                best_model = model
                best_result = dev_score
        return best_model
