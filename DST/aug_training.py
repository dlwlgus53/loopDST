import sys
sys.path.append('../')
from ..data_augment3.augment import log_setting, get_will_change_item, generate_new_text
import torch
import random
import copy
import numpy as np
import json
import ontology
import random
import os
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from dataclass_part import DSTMultiWozData
from trainer_part import train, evaluate



all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class Aug_training:
    def __init__(self,aug_num):
        self.log = log_setting()
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
        self.aug_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.aug_config = RobertaConfig()
        self.aug_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(DEVICE)
        self.aug_num

    
    def _makedirs(self, path): 
        try: 
            os.makedirs(path) 
        except OSError: 
            if not os.path.isdir(path): 
                raise

    def augment(self, raw_data, labeled_data, change_rate, device, aug_batch_size = 10):
        raw_data = self.data.replace_label(raw_data, labeled_data)
        dial_turn_id_list, tokenized_masked_list = get_will_change_item(raw_data, self.aug_tokenizer, change_rate)
        generated_dict= generate_new_text(self.aug_model, dial_turn_id_list, tokenized_masked_list, aug_batch_size, device)
        raw_data_similar = []
        for dial_idx, dial in enumerate(raw_data):
            if dial_idx%30 == 0 and dial_idx !=0: self.log.info(f'saving dials {dial_idx}/{len(raw_data)} done')
            for n in range(self.aug_num):
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
    
    def train(self, log, args, data, model, epoch, optimizer, scheduler, cuda_available, device, specify_adafactor_lr):
        best_result = 0
        best_model = None
        for epoch in range(epoch):
            train_loss = train(args,model,optimizer, scheduler,specify_adafactor_lr, data,log, cuda_available, device)
            _, dev_score = evaluate(args,model,data,log, cuda_available, device)
            if dev_score > best_result:
                best_model = model
                best_result = dev_score
        return best_model
