import sys
sys.path.append('../../../../')
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
    def __init__(self,aug_method, aug_num, change_rate, data, device, log,log_interval, batch_size):
        
        if aug_method == 2:
            from data_augment2.augment import log_setting, get_generated_dict
        if aug_method ==3:
            from data_augment3.augment import log_setting, get_generated_dict
            
        log_setting("aug_log")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
        self.aug_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.aug_config = RobertaConfig()
        self.aug_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(DEVICE)
        self.aug_num = aug_num
        self.change_rate = change_rate
        self.log = log
        self.get_generated_dict = get_generated_dict
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.log_interval = log_interval
    
    def _makedirs(self, path): 
        try: 
            os.makedirs(path) 
        except OSError: 
            if not os.path.isdir(path): 
                raise

    def augment(self):
        
        raw_data = self.data.replace_label(self.data.train_raw_data, self.data.labeled_data)
        generated_dict = self.get_generated_dict(raw_data, self.aug_tokenizer, self.aug_model, self.change_rate, \
            self.aug_num, self.batch_size, self.device, self.log,self.log_interval)
        
        raw_data_similar = []
        for _, dial in enumerate(raw_data):
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
            train_loss = train(args,model,optimizer, scheduler, self.data,self.log, cuda_available = True,  device = device, mode = 'train_aug')
            _, dev_score = evaluate(args,model,self.data,self.log, cuda_available = True, device = device,  mode = 'dev_aug')
            self.log.info(f"Aug JGA : {dev_score:.2f}")
            if dev_score > best_result:
                best_model = model
                best_result = dev_score
        return best_model
