from cProfile import label
import pdb
import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import random
from torch.nn.utils import rnn
import logging
import logging.handlers
import copy
import time
from transformers import T5Config
import json


all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']
sos_eos_tokens = ['<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<go_d>','<eos_d>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', 
                '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>']
class NLIdata:
    def __init__(self, model_name, tokenizer, data_path_prefix,  ckpt_save_path, \
        add_prefix=True, add_special_decoder_token=True):

        self.log = self.set_log("./log.txt")
        self.add_prefix = add_prefix
        assert self.add_prefix in [True, False]
        self.add_special_decoder_token = add_special_decoder_token
        assert self.add_special_decoder_token in [True, False]
        self.tokenizer = tokenizer
        self.log.info ('Original Tokenizer Size is %d' % len(self.tokenizer))
        self.special_token_list = self.add_sepcial_tokens()
        self.log.info ('Tokenizer Size after extension is %d' % len(self.tokenizer))
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
        self.model_name = model_name
        assert self.model_name.startswith('t5')
        t5config = T5Config.from_pretrained(model_name)
        self.bos_token_id = t5config.decoder_start_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token = self.tokenizer.convert_ids_to_tokens([self.bos_token_id])[0]
        self.eos_token = self.tokenizer.convert_ids_to_tokens([self.eos_token_id])[0]
        self.log.info ('bos token is {}, eos token is {}'.format(self.bos_token, self.eos_token))
        self.all_sos_token_id_list = []
        for token in all_sos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            self.all_sos_token_id_list.append(one_id)
        self.all_eos_token_id_list = []
        for token in all_eos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            self.all_eos_token_id_list.append(one_id)
            
        # Change to propoer one..    
        # 이 대화에서 호텔, 정보가 일치하는가?
        # yes/no
        
        if self.add_prefix:
            bs_prefix_text = 'Do two sentences have the same DST results? (same or different)'
            self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
        else:
            self.bs_prefix_id = []


        
        self.data_path_prefix = data_path_prefix 
        self.ckpt_save_path = ckpt_save_path 
        
        similar_data_path = data_path_prefix + 'similar.json'
        different_data_path = data_path_prefix + 'different.json'
        
        

        with open(similar_data_path) as f:
            similar_data = json.load(f)
            
        with open(different_data_path) as f:
            different_data = json.load(f)
            
        merged_data=self.merge_files(similar_data, different_data, shuffle = True)
        raw_train_data, raw_dev_data, raw_test_data = self.train_dev_test_split(merged_data)    
        self.train_data = self.tokenize_and_to_id(raw_train_data) 
        self.dev_data = self.tokenize_and_to_id(raw_dev_data) 
        self.test_data = self.tokenize_and_to_id(raw_test_data) 
        
        self.train_num = len(self.train_data)
        self.dev_num = len(self.dev_data)
        self.test_num = len(self.test_data)

    def train_dev_test_split(self, merged_data):
        len_data = len(merged_data)
        part1 = int(len_data*0.8)
        part2 = int(len_data*0.9)
        return merged_data[:part1], merged_data[part1:part2], merged_data[part2:]
    
    
    def merge_files(self, similar_data, different_data, shuffle = True):
        merged_list = []
        for dial in similar_data:
            for turn in dial:
                if turn['user_similar'] == '<sos_u> -1 <eos_u>':continue
                user = turn['user']
                user_similar = turn['user_similar']
                label = 'same'
                merged_list.append(
                    {'user1' : user,
                     'user2' : user_similar,
                     'label' : label}
                )
                
        for dial in different_data:
            for turn in dial:
                if turn['user_different'] == '<sos_u> -1 <eos_u>':continue
                user = turn['user']
                user_different = turn['user_different']
                label = 'different'
                merged_list.append(
                    {'user1' : user,
                    'user2' : user_different,
                    'label' : label
                    }
                )
                
        if shuffle:
            random.shuffle(merged_list)
            
        return merged_list
                
    def set_log(self,log_path):
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
    
    def replace_sos_eos_token_id(self, token_id_list):
        if self.add_special_decoder_token: # if adding special decoder tokens, then no replacement
            sos_token_id_list = []
            eos_token_id_list = []
        else:
            sos_token_id_list = self.all_sos_token_id_list
            eos_token_id_list = self.all_eos_token_id_list

        res_token_id_list = []
        for one_id in token_id_list:
            if one_id in sos_token_id_list:
                res_token_id_list.append(self.bos_token_id)
            elif one_id in eos_token_id_list:
                res_token_id_list.append(self.eos_token_id)
            else:
                res_token_id_list.append(one_id)
        return res_token_id_list

    def tokenize_and_to_id(self, raw_data):
        all_session_list = []
        for turn in raw_data: 
            one_turn_dict = {}
            for key in turn:
            
                value_text = turn[key]
                value_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_text))
                value_id = self.replace_sos_eos_token_id(value_id)
                one_turn_dict[key] = value_id
            one_turn_dict['input'] = self.bs_prefix_id + one_turn_dict['user1'] +  one_turn_dict['user2'] 
            all_session_list.append(one_turn_dict)
        return all_session_list


    def tokenized_decode(self, token_id_list):
        pred_tokens = self.tokenizer.convert_ids_to_tokens(token_id_list)
        res_text = ''
        curr_list = []
        for token in pred_tokens:
            if token in self.special_token_list + ['<s>', '</s>', '<pad>']:
                if len(curr_list) == 0:
                    res_text += ' ' + token + ' '
                else:
                    curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
                    res_text = res_text + ' ' + curr_res + ' ' + token + ' '
                    curr_list = []
            else:
                curr_list.append(token)
        if len(curr_list) > 0:
            curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
            res_text = res_text + ' ' + curr_res + ' '
        res_text_list = res_text.strip().split()
        res_text = ' '.join(res_text_list).strip()
        return res_text

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        special_tokens = sos_eos_tokens

        self.tokenizer.add_tokens(special_tokens)
        return special_tokens


    def build_iterator(self, batch_size, mode ):
        batch_list= self.get_batches(batch_size, mode)
        for batch in batch_list:
            yield batch
            
    def get_batches(self, batch_size, mode): # 여기서 전부다 합친다음 배치사이즈만큼 미리 쪼갠다!
        batch_list = []

        if mode == 'train':
            all_data_list = self.train_data
        elif mode == 'valid':
            all_data_list = self.dev_data
        elif mode == 'test':
            all_data_list = self.test_data

        all_input_data_list, all_output_data_list = [], []

        for item in all_data_list:
            all_input_data_list.append(item['input'])
            all_output_data_list.append(item['label'])
        
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
            
            
            
        return batch_list
            


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
        return label

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        label, _ = self.pad_batch(batch_output_id_list)
        return batch_src_tensor, batch_src_mask, label

    def remove_sos_eos_token(self, text):
        token_list = text.split()
        res_list = []
        for token in token_list:
            if token == '<_PAD_>' or token.startswith('<eos_') or token.startswith('<sos_') or token in [self.bos_token, self.eos_token]:
                continue
            else:
                res_list.append(token)
        return ' '.join(res_list).strip()

    def parse_id_to_text(self, id_list):
        res_text = self.tokenized_decode(id_list)
        res_text = self.remove_sos_eos_token(res_text)
        return res_text

    def parse_one_eva_instance(self, one_instance):
        '''
            example data instance:
                {'dial_id': 'sng0547',
                 'turn_num': 0,
                 'user': 'i am looking for a high end indian restaurant, are there any in town?',
                 'bspn_gen': '[restaurant] food indian pricerange expensive',
                 'bsdx': '[restaurant] food pricerange',
                 'resp_gen': 'there are [value_choice] . what area of town would you like?',
                 'resp': 'there are [value_choice] [value_price] [value_food] restaurant -s in cambridge. is there an area of town that you prefer?',
                 'bspn': '[restaurant] food indian pricerange expensive',
                 'pointer': 'restaurant: >3; '}
            input_contain_db: whether input contain db result
            ref_db: if input contain db, whether using the reference db result
        '''
        res_dict = {}
        res_dict['dial_id'] = one_instance['dial_id']
        res_dict['turn_num'] = one_instance['turn_num']
        res_dict['user'] = self.parse_id_to_text(one_instance['user'])
        res_dict['bspn'] = self.parse_id_to_text(one_instance['bspn'])
        res_dict['bsdx'] = self.parse_id_to_text(one_instance['bsdx'])
        res_dict['bspn_reform'] = self.parse_id_to_text(one_instance['bspn_reform'])
        res_dict['bsdx_reform'] = self.parse_id_to_text(one_instance['bsdx_reform'])
        previous_context = one_instance['prev_context']
        curr_user_input = one_instance['user']

        # belief state setup
        res_dict['bspn_gen'] = ''
        bs_input_id_list = previous_context + curr_user_input
        bs_input_id_list = self.bs_prefix_id + [self.sos_context_token_id] + bs_input_id_list[-900:] + [self.eos_context_token_id]
        return bs_input_id_list, res_dict

