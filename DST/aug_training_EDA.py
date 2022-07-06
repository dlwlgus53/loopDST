import sys
sys.path.append('../../../../')
import torch
import copy
import json
import ontology
import random
import os
import pdb
from dataclass_part import DSTMultiWozData
from trainer_part import train, evaluate



all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class Aug_training:
    def __init__(self,aug_method, aug_num, data, log, log_interval, use_dev_aug):
        
        if aug_method == 11:
            from data_augment11.augment import log_setting, get_generated_dict
        if aug_method == 12:
            from data_augment12.augment import log_setting, get_generated_dict

        log_setting("aug_log")
        self.aug_num = aug_num
        self.log = log
        self.get_generated_dict = get_generated_dict
        self.data = data
        self.use_dev_aug= use_dev_aug
        
        
    
    def _makedirs(self, path): 
        try: 
            os.makedirs(path) 
        except OSError: 
            if not os.path.isdir(path): 
                raise

    def augment(self):
        raw_data = self.data.replace_label(self.data.train_raw_data, self.data.labeled_data)
        generated_dict = self.get_generated_dict(raw_data, self.aug_num, self.log)
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
                        # 이건 어차피 안바뀌는거니까 상관이 없지
                    else:
                        pass
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
        device = torch.device('cuda')
        best_result = 0
        best_model = None
        self.data.set_train_aug(train_data)
        not_progress = 0
        self.data.set_eval_aug(dev_data)
        for epoch in range(epoch):
            not_progress +=1
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