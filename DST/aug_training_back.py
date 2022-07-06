# augmentation for backtranslation
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
from transformers import MarianMTModel, MarianTokenizer
from dataclass_part import DSTMultiWozData
from trainer_part import train, evaluate
from data_augment10.augment import log_setting, get_generated_dict


all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class Aug_training:
    def __init__(self,aug_method, aug_num, change_rate, data, device, log,log_interval, batch_size, use_dev_aug):
            
        log_setting("aug_log")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
        model_name1 = 'Helsinki-NLP/opus-mt-en-fr'
        model_name2 = 'Helsinki-NLP/opus-mt-fr-en'

        self.tokenizer1 = MarianTokenizer.from_pretrained(model_name1)
        self.model1 = MarianMTModel.from_pretrained(model_name1).to(DEVICE)

        self.tokenizer2 = MarianTokenizer.from_pretrained(model_name2)
        self.model2 = MarianMTModel.from_pretrained(model_name2).to(DEVICE)
        
        self.aug_num = aug_num
        self.change_rate = change_rate
        self.log = log
        self.get_generated_dict = get_generated_dict
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.use_dev_aug= use_dev_aug
        
    
    def _makedirs(self, path): 
        try: 
            os.makedirs(path) 
        except OSError: 
            if not os.path.isdir(path): 
                raise

    def augment(self):
        
        raw_data = self.data.replace_label(self.data.train_raw_data, self.data.labeled_data)
        
        generated_dict = get_generated_dict(raw_data = raw_data, tokenizer1 = self.tokenizer1,
                                            tokenizer2 = self.tokenizer2,  model1 = self.model1,
                                            model2 = self.model2, aug_num = self.aug_num,
                                            batch_size = self.batch_size, device = 'cuda', 
                                            log = self.log)
        
        
        raw_data_similar = []
        for _, dial in enumerate(raw_data):
            for n in range(self.aug_num):
                similar_dial = []
                for turn in dial:
                    idx = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                    similar_turn = copy.deepcopy(turn)
                    similar_turn['dial_id'] += f'_v{str(n)}'
                    similar_turn['user'] = generated_dict[idx]['text']
                    similar_dial.append(similar_turn)
                raw_data_similar.append(similar_dial)
        if self.use_dev_aug:
            train = raw_data_similar[:int(len(raw_data_similar) * 0.9):]
            dev = raw_data_similar[int(len(raw_data_similar) * 0.9):]
            return train, dev
        else:
            train = raw_data_similar
            return train, []
    
    def train(self, args, train_data, dev_data, model, epoch, optimizer, scheduler):
        device = torch.device(self.device)
        best_result = 0
        best_model = None
        not_progress = 0
        self.data.set_train_aug(train_data)
        self.data.set_eval_aug(dev_data)
        for epoch in range(epoch):
            not_progress += 1
            if not_progress > args.patient:
                self.log.info(f"early stopping in aug training epoch : {epoch}")
                break
            train_loss = train(args,model,optimizer, scheduler, self.data,self.log, cuda_available = True,  device = device, mode = 'train_aug')
            if self.use_dev_aug:
                _, dev_score = evaluate(args,model,self.data,self.log, cuda_available = True, device = device,  mode = 'dev_aug')
            else:
                _, dev_score = evaluate(args,model,self.data,self.log, cuda_available = True, device = device,  mode = 'dev_loop')
            self.log.info(f"Aug JGA : {dev_score:.2f}")
            if dev_score > best_result:
                not_progress =0
                best_model = model
                best_result = dev_score
        return best_model