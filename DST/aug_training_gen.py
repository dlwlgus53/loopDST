import sys
sys.path.append('../../../../')
sys.path.append('../../../../../')

import torch
import random
import copy
import numpy as np
import random
import os
from dataclass_part import DSTMultiWozData
from trainer_part import train, evaluate
from data_augment5.augment import log_setting, get_generated_dict, load_model
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor, T5Config

all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class Aug_training:
    def __init__(self,aug_method, aug_num, change_rate, data, device, log,log_interval, batch_size, model_path = None):
        log_setting("aug_log")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = load_model(model_path, device, multi_gpu_training = True)
        self.aug_num = aug_num
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
        generated_dict = self.get_generated_dict(raw_data, self.tokenizer, self.model, self.aug_num,'cuda', self.log)
        raw_data_similar = []
        for dial_idx, dial in enumerate(raw_data):
            for n in range(self.aug_num):
                similar_dial = []
                for turn in dial:
                    idx = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                    similar_turn = copy.deepcopy(turn)
                    if idx in generated_dict:
                        similar_turn['dial_id'] += f'_v{str(n)}'
                        similar_turn['user'] = generated_dict[idx]['text']
                        similar_turn['bspn'] = generated_dict[idx]['bspn']
                        similar_dial.append(similar_turn)
                    else:
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


