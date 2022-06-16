import pdb
import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import random
import logging
import logging.handlers
import data_augment5.model_train.ontology as ontology
import copy
import time
from transformers import T5Config


class Generate_dataclass:
    def __init__(self, model_name, tokenizer, data_path_prefix, ckpt_save_path, log_path, init_label_path, shuffle_mode='shuffle_session_level', 
        debugging = False):
        
        # add special words to tokenizer
        
        self.log = self.log_setting(log_path)
        self.tokenizer = tokenizer
        self.debugging = debugging
        train_json_path = data_path_prefix + '/multiwoz-fine-processed-tenpercent.json' 
        
        if self.debugging : 
            small_path = data_path_prefix + '/multiwoz-fine-processed-small_dev.json' 
            train_json_path = dev_json_path = test_json_path = small_path

        with open(train_json_path) as f:
            self.train_raw_data = json.load(f)


    def log_setting(self, log_path):
        log = logging.getLogger('log in data')
        log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')

        fileHandler = logging.FileHandler(log_path)
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        log.addHandler(fileHandler)
        log.addHandler(streamHandler)
        return log
    
    
    
    def bspn_split_this(self,prev_bspn_dict, bspn_dict):
        this_bspn = {}
        
        for domain in bspn_dict:
            if domain not in prev_bspn_dict:
                this_bspn[domain] = bspn_dict[domain]
                continue
            for slot in bspn_dict[domain]:
                value = bspn_dict[domain][slot]
                if slot in prev_bspn_dict[domain] and value == prev_bspn_dict[domain][slot]:
                    pass
                else:
                    if domain in this_bspn:
                        this_bspn[domain][slot] = value
                    else:
                        this_bspn[domain] = {}
                        this_bspn[domain][slot] = value
                        
        return  this_bspn

    def bspn_to_dict(self, bspn):
        slot_info = ontology.informable_slots
        bspn =bspn.replace("<sos_b> ",'').replace("<eos_b>","").split(" ")
        bspn_dict = {}
        
        domain = ''
        slot = ''
        try:
            for word in bspn:
                if len(word)==0:continue
                if word[0] == '[':
                    domain = word[1:-1]
                    slot = ''
                    bspn_dict[domain] = {}
                elif word in slot_info[domain]:
                    slot = word
                    bspn_dict[domain][slot] = ''
                else:
                    if len(bspn_dict[domain][slot]) == 0:
                        bspn_dict[domain][slot] = word
                    else:
                        bspn_dict[domain][slot] += (' ' + word)
        except:
            pdb.set_trace()            
                
        return bspn_dict
    
    def dict_to_bspn(self,bspn_dict):
        bspn = '<sos_b>'
        
        for domain in bspn_dict:
            bspn += (" [" + domain + "]")
            for slot in bspn_dict[domain]:
                bspn += (' ' + slot + ' ' + bspn_dict[domain][slot])
        bspn += ' <eos_b>'
        return bspn


    def tokenize_raw_data(self, raw_data_list): # TODO also get labeld data list and answer
        data_num = len(raw_data_list)
        all_session_list = []
        for idx in range(data_num):
            one_sess_list = []
            for turn in raw_data_list[idx]: 
                one_turn_dict = {}
                for key in turn:
                    if key in ['dial_id', 'turn_num','user', 'resp', 'bspn']:
                        one_turn_dict[key] = turn[key]
                        if key == 'bspn': # 이전것을 넣어야 할 것 같은데?!
                            if len(one_sess_list) == 0:
                                one_turn_dict['prev_bspn'] = ''
                                one_turn_dict['this_bspn'] = turn[key]
                            else:
                                prev_turn_bspn_dict = self.bspn_to_dict(one_sess_list[-1]['bspn'])
                                turn_bspn_dict = self.bspn_to_dict(turn[key])
                                this_bspn_dict = self.bspn_split_this(prev_turn_bspn_dict, turn_bspn_dict)
                                one_turn_dict['prev_bspn'] = one_sess_list[-1]['bspn']
                                one_turn_dict['this_bspn'] = self.dict_to_bspn(this_bspn_dict)

                    # elif key in  ['user', 'resp', 'bspn']:
                    #     value_text = turn[key]
                    #     value_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_text))
                    #     one_turn_dict[key] = value_id
                        
                    #     if key == 'bspn':
                    #         value_text = turn[key] # change to only this belief
                    #         value_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_text))
                    #         one_turn_dict['this_bspn'] = value_id
                            
                one_sess_list.append(one_turn_dict)
            all_session_list.append(one_sess_list)
        assert len(all_session_list) == len(raw_data_list)
        return all_session_list
    
    def flatten_data(self, data):
        # prefix = self.tokenizer("generate the user utterance : ")['input_ids']
        # context_prefix = self.tokenizer("<context>")['input_ids']
        # bspn_prefix =  self.tokenizer("<bspn>")['input_ids']
        # this_turn_prefix =self.tokenizer("<this_bspn>")['input_ids']
         
        data_list = []
        for session in data: # session is dial
            one_dial_id = session[0]['dial_id']
            turn_num = len(session)
            previous_context = '' # previous context contains all previous user input and system response
            for turn_id in range(turn_num):
                curr_turn = session[turn_id]
                assert curr_turn['turn_num'] == turn_id # the turns should be arranged in order
                # TODO : change this part to idx
                # generate_input = context_prefix + previous_context[-900:] + bspn_prefix + curr_turn['bspn'] + this_turn_prefix + curr_turn['this_bspn']
                generate_input ="generate the user utterance : " + previous_context[-900:] + "<prev_bspn>" + curr_turn['prev_bspn'] + "<this_bspn>" + curr_turn['this_bspn']
                data_list.append({'dial_id': one_dial_id,
                    'turn_num': turn_id,
                    # 'input': prefix + generate_input,
                    'input':generate_input,
                    'output' : curr_turn['user']
                    })
                
                previous_context = previous_context + curr_turn['user'] + curr_turn['resp']
        return data_list


    def make_data_list(self, raw_data):
        data_id_list = self.tokenize_raw_data(raw_data) # give labled data list too
        data_list = self.flatten_data(data_id_list)
        return data_list

    def filter_data(self, raw, filter, use_label):
        new_data = []
        if use_label == False:
            for dial in raw:
                dial_turn_key = '[d]'+dial[0]['dial_id'] + '[t]' + str(0)
                if dial_turn_key not in filter.keys():
                    new_data.append(dial)
        return new_data
    
    def replace_label(self, raw, label):
        new_raw = copy.deepcopy(raw)
        for dial in new_raw:
            for turn in dial:
                dial_turn_key = '[d]'+turn['dial_id'] + '[t]' + str(turn['turn_num'])
                if dial_turn_key in label:
                    turn['bspn'] = label[dial_turn_key]
        return new_raw
        
    def get_filtered_batches(self, batch_size, mode): 
        batch_list = []
        idx_list = []
        if mode == 'train':
            raw_data = self.train_raw_data[:int(len(self.train_raw_data) * 0.9)]
            self.train_data_list = self.make_data_list(raw_data) # make dataset with labeled data
            all_data_list = self.train_data_list 
        elif mode == 'dev':
            raw_data= self.train_raw_data[int(len(self.train_raw_data) * 0.9):]
            # raw_data = self.filter_data(self.train_raw_data, self.labeled_data, use_label = False)
            self.dev_data_list = self.make_data_list(raw_data) # make dataset with labeled data
            all_data_list = self.dev_data_list
        # elif mode == 'test':
        #     raw_data = self.replace_label(self.train_aug_raw_data, self.labeled_data)
        #     self.train_aug_data_list = self.make_data_list(raw_data) # make dataset with labeled data
        #     all_data_list = self.train_aug_data_list
        else:
            raise Exception('Wrong Mode!!!')
        
        all_input_data_list, all_output_data_list = [], []
        for item in all_data_list:  
            all_input_data_list.append(item['input'])
            all_output_data_list.append(item['output'])
        data_num = len(all_input_data_list)
        batch_num = int(data_num/batch_size) + 1

        for i in range(batch_num):
            start_idx, end_idx = i*batch_size, (i+1)*batch_size
            if start_idx > data_num - 1:
                break
            end_idx = min(end_idx, data_num - 1)
            one_input_batch_list, one_output_batch_list, one_index_list = [], [], []
            for idx in range(start_idx, end_idx):
                one_input_batch_list.append(all_input_data_list[idx])
                one_output_batch_list.append(all_output_data_list[idx])
                
            one_batch = [one_input_batch_list, one_output_batch_list]
            batch_list.append(one_batch)
        out_str = f'Overall Number of datapoints of {mode} is \
            {str(data_num)}  and batches is {str(len(batch_list))}'
            
        self.log.info (out_str)
        
        return batch_list
            
    def build_iterator(self, batch_size, mode):
        batch_list= self.get_filtered_batches(batch_size, mode)
        for batch in batch_list:
            yield batch



    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask 

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list) # padded target sequence
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        # batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        # batch_input, batch_labels = self.process_output(batch_output_id_list)
        # batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        
        batch_input = self.tokenizer(batch_input_id_list, padding = True, truncation = True)
        batch_output = self.tokenizer(batch_output_id_list, padding = True, truncation = True)
        
        source_input = torch.tensor(batch_input['input_ids'])
        source_mask = torch.tensor(batch_input['attention_mask'])
        target_input = torch.tensor(batch_output['input_ids'])
        target_mask = torch.tensor(batch_output['attention_mask'])
        return source_input, source_mask, target_input, target_mask
