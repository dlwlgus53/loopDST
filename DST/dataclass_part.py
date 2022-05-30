import pdb
import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import ontology
import random
import logging
import logging.handlers
import copy
import time
from transformers import T5Config

all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class DSTMultiWozData:
    def __init__(self, model_name, tokenizer, data_path_prefix, ckpt_save_path, log_path, init_label_path, shuffle_mode='shuffle_session_level', 
        debugging = False):
        self.log = self.log_setting(log_path)
        self.tokenizer = tokenizer
        self.log.info ('Original Tokenizer Size is %d' % len(self.tokenizer))
        self.special_token_list = self.add_special_tokens()
        self.log.info ('Tokenizer Size after extension is %d' % len(self.tokenizer))
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]

        # initialize bos_token_id, eos_token_id
        self.model_name = model_name
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

        bs_prefix_text = 'translate dialogue to belief state:'
        self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
        self.data_path_prefix = data_path_prefix 
        self.ckpt_save_path = ckpt_save_path 
        self.debugging = debugging
        
        init_labeled_json_path = init_label_path + '/labeled_init.json'
        self.log.info (f"load initial labeled data from {init_labeled_json_path}")
        with open(init_labeled_json_path) as f:
            self.init_labeled_data = json.load(f)
            
        self.labeled_data = copy.deepcopy(self.init_labeled_data)
                            
        train_json_path = data_path_prefix + '/multiwoz-fine-processed-train-1.json' 
        dev_json_path = data_path_prefix + '/multiwoz-fine-processed-dev.json'
        test_json_path = data_path_prefix + '/multiwoz-fine-processed-test.json'
        
        if self.debugging : 
            small_path = data_path_prefix + '/multiwoz-fine-processed-small-1.json' 
            train_json_path = dev_json_path = test_json_path = small_path

        with open(train_json_path) as f:
            self.train_raw_data = json.load(f)
        with open(dev_json_path) as f:
            dev_raw_data = json.load(f)
        with open(test_json_path) as f:
            test_raw_data = json.load(f)

        self.train_data_list, self.tagging_data_list = [],[]
        self.dev_data_list = self.make_data_list(raw_data = dev_raw_data)
        self.test_data_list = self.make_data_list(raw_data = test_raw_data)

        self.log.info (' dev turn number is %d, test turn number is %d ' % \
            ( len(self.dev_data_list), len(self.test_data_list)))

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

    def replace_sos_eos_token_id(self, token_id_list):
        sos_token_id_list = []
        eos_token_id_list = []
        res_token_id_list = []
        
        for one_id in token_id_list:
            if one_id in sos_token_id_list:
                res_token_id_list.append(self.bos_token_id)
            elif one_id in eos_token_id_list:
                res_token_id_list.append(self.eos_token_id)
            else:
                res_token_id_list.append(one_id)
        return res_token_id_list

    def tokenize_raw_data(self, raw_data_list): # TODO also get labeld data list and answer
        data_num = len(raw_data_list)
        all_session_list = []
        for idx in range(data_num):
            one_sess_list = []
            for turn in raw_data_list[idx]: 
                one_turn_dict = {}
                for key in turn:
                    if key in ['dial_id', 'pointer', 'turn_domain', 'turn_num', 'aspn', 'dspn', 'aspn_reform', 'db']:
                        one_turn_dict[key] = turn[key]
                    else: 
                        value_text = turn[key]
                        value_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_text))
                        value_id = self.replace_sos_eos_token_id(value_id)
                        one_turn_dict[key] = value_id
                one_sess_list.append(one_turn_dict)
            all_session_list.append(one_sess_list)
        assert len(all_session_list) == len(raw_data_list)
        return all_session_list
    
    # def get_init_data(self):
    #     return self.init_labeled_data
    
    def update_labeled_data(self,labeled_data):
        self.log.info("Update the labeled_data")
        self.labeled_data = labeled_data

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

    def add_special_tokens(self):
        special_tokens = []
        special_tokens = ontology.sos_eos_tokens
        self.tokenizer.add_tokens(special_tokens)
        return special_tokens

    def flatten_data(self, data):
        data_list = []
        for session in data: # session is dial
            one_dial_id = session[0]['dial_id']
            turn_num = len(session)
            previous_context = [] # previous context contains all previous user input and system response
            for turn_id in range(turn_num):
                curr_turn = session[turn_id]
                assert curr_turn['turn_num'] == turn_id # the turns should be arranged in order
                curr_user_input = curr_turn['user']
                curr_sys_resp = curr_turn['resp']
                curr_bspn = curr_turn['bspn']
                bs_input = previous_context + curr_user_input
                bs_input = self.bs_prefix_id + [self.sos_context_token_id] + bs_input[-900:] + [self.eos_context_token_id]
                bs_output = curr_bspn
                data_list.append({'dial_id': one_dial_id,
                    'turn_num': turn_id,
                    'prev_context':previous_context,
                    'user': curr_turn['user'],
                    'usdx': curr_turn['usdx'],
                    'resp': curr_sys_resp,
                    'bspn': curr_turn['bspn'],
                    'bspn_reform': curr_turn['bspn_reform'],
                    'bsdx': curr_turn['bsdx'],
                    'bsdx_reform': curr_turn['bsdx_reform'],
                    'bs_input': bs_input,
                    'bs_output': bs_output
                    })
                # update context for next turn
                previous_context = previous_context + curr_user_input + curr_sys_resp
        
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
        if mode == 'train_loop':
            raw_data = self.replace_label(self.train_raw_data, self.labeled_data)
            self.train_data_list = self.make_data_list(raw_data) # make dataset with labeled data
            all_data_list = self.train_data_list 
        elif mode == 'tagging':
            raw_data= copy.deepcopy(self.train_raw_data)
            # raw_data = self.filter_data(self.train_raw_data, self.labeled_data, use_label = False)
            self.tagging_data_list = self.make_data_list(raw_data) # make dataset with labeled data
            all_data_list = self.tagging_data_list
        elif mode == 'train_aug':
            raw_data = self.replace_label(self.train_aug_raw_data, self.labeled_data)
            self.train_aug_data_list = self.make_data_list(raw_data) # make dataset with labeled data
            all_data_list = self.train_aug_data_list
            
        else:
            raise Exception('Wrong Mode!!!')
        all_input_data_list, all_output_data_list, all_index_list = [], [], []
        for item in all_data_list:  
            dial_turn_key = '[d]'+item['dial_id'] + '[t]' + str(item['turn_num'])
            if mode == 'tagging' and dial_turn_key in self.labeled_data : continue
            if mode == 'train' and dial_turn_key not in self.labeled_data : continue
            if mode == 'train_aug':
                dial_turn_key =  '[d]'+item['dial_id'].split("_")[0] + '[t]' + str(item['turn_num'])
                if dial_turn_key not in self.labeled_data : continue
            all_input_data_list.append(item['bs_input'])
            all_output_data_list.append(item['bs_output'])
            all_index_list.append(dial_turn_key)
        
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
                one_index_list.append(all_index_list[idx])
                
            one_batch = [one_input_batch_list, one_output_batch_list]
            one_idx = one_index_list
            batch_list.append(one_batch)
            idx_list.append(one_idx)
        # Train aug에서 애매한 이유 : dev로 빠지기 때문
        out_str = 'Overall Number of datapoints of ' + mode + ' is ' + str(data_num) + \
        ' and batches is ' + str(len(batch_list))
        self.log.info (out_str)
        
        return batch_list, idx_list
            
    def build_iterator(self, batch_size, mode ):
        batch_list, idx_list = self.get_filtered_batches(batch_size, mode)
        for batch, idx in zip(batch_list, idx_list):
            yield batch, idx



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
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels

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

################################## need only for evaluation ###########################################
    def parse_one_eva_instance(self, one_instance):
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

        res_dict['bspn_gen'] = ''
        bs_input_id_list = previous_context + curr_user_input
        bs_input_id_list = self.bs_prefix_id + [self.sos_context_token_id] + bs_input_id_list[-900:] + [self.eos_context_token_id]
        return bs_input_id_list, res_dict

    def build_evaluation_batch_list(self, all_data_list, batch_size):
        data_num = len(all_data_list)
        batch_num = int(data_num/batch_size) + 1
        batch_list = []
        for i in range(batch_num):
            start_idx, end_idx = i*batch_size, (i+1)*batch_size
            if start_idx > data_num - 1:
                break
            # end_idx = min(end_idx, data_num - 1)
            end_idx = min(end_idx, data_num)
            one_batch_list = []
            for idx in range(start_idx, end_idx):
                one_batch_list.append(all_data_list[idx])
            if len(one_batch_list) == 0: 
                pass
            else:
                batch_list.append(one_batch_list)
        return batch_list

    def build_all_evaluation_batch_list(self, eva_batch_size, eva_mode):
        if eva_mode == 'dev_loop':
            data_list = self.dev_data_list
        elif eva_mode == 'test':
            data_list = self.test_data_list
        elif eva_mode == 'dev_aug':
            data_list = self.dev_aug_data_list
        else:
            raise Exception('Wrong Evaluation Mode!!!')
        
        all_bs_input_id_list, all_parse_dict_list = [], []
        for _, item in enumerate(data_list):
            one_bs_input_id_list, one_parse_dict = self.parse_one_eva_instance(item)
            all_bs_input_id_list.append(one_bs_input_id_list)
            all_parse_dict_list.append(one_parse_dict)
        assert len(all_bs_input_id_list) == len(all_parse_dict_list)
        bs_batch_list = self.build_evaluation_batch_list(all_bs_input_id_list, eva_batch_size)
        parse_dict_batch_list = self.build_evaluation_batch_list(all_parse_dict_list, eva_batch_size)

        batch_num = len(bs_batch_list)
        final_batch_list = []
        
        for idx in range(batch_num):
            one_final_batch = [bs_batch_list[idx], parse_dict_batch_list[idx]]
            if len(bs_batch_list[idx]) == 0: 
                continue
            else:
                final_batch_list.append(one_final_batch)
        return final_batch_list


############################### aug #################################
    def set_train_aug(self, train_aug_data):
        # 원래 데이터의 2배가 아닌 이유 : 10% 는 DEV로 사용하기 때문
        self.train_aug_raw_data = train_aug_data

    def set_eval_aug(self, dev_aug_data):
        self.dev_aug_data_list = self.make_data_list(raw_data = dev_aug_data)