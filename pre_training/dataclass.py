from cProfile import label
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
from torch.nn.utils import rnn
import logging
import logging.handlers
import copy
import time



class DSTMultiWozData:
    def __init__(self, model_name, tokenizer, data_path_prefix, ckpt_save_path, log_path, tagging_all = False, shuffle_mode='shuffle_session_level', 
        add_prefix=True, add_special_decoder_token=True, train_data_ratio=1.0,  debugging = False):
        

        log = logging.getLogger('log in data')
        log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')

        fileHandler = logging.FileHandler(log_path)
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        log.addHandler(fileHandler)
        log.addHandler(streamHandler)
        
        self.log = log
        self.tagging_all = tagging_all
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
        from transformers import T5Config
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
            
        # TODO : shoud I change?
        if self.add_prefix:
            bs_prefix_text = 'translate dialogue to belief state:'
            self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
        else:
            self.bs_prefix_id = []
        
        self.data_path_prefix = data_path_prefix 
        self.ckpt_save_path = ckpt_save_path 
        import json
        self.debugging = debugging
        
        similar_data_path = data_path_prefix + '/labeled_init.json'
        different_data_path = data_path_prefix + '/labeled_init.json'

        # 두 파일을 합치고
        # 적절히 섞는다.
        # Train, valid, test 만든다.
                                
        if data_mode == 'train':
            train_data_id_list = self.tokenize_raw_data(train_raw_data) # give labled data list too
            self.train_data_list = self.flatten_data(train_data_id_list)
            # record training data
            self.train_id2session_dict = {}
            self.train_dial_id_list = []
            for item in self.train_data_list:
                one_item_id = item['dial_id']
                try:
                    self.train_id2session_dict[one_item_id].append(item)
                except KeyError:
                    self.train_dial_id_list.append(one_item_id)
                    self.train_id2session_dict[one_item_id] = [item]
            assert len(self.train_dial_id_list) == len(self.train_id2session_dict)
            self.train_num = len(self.train_data_list) 
        elif data_mode == 'test':
            train_raw_data = []
        else:
            raise Exception('Wrong Data Mode!!!')

        dev_json_path = data_path_prefix + '/multiwoz-fine-processed-dev.json'
        if self.debugging : dev_json_path = data_path_prefix + '/multiwoz-fine-processed-small_dev.json' 
        
        with open(dev_json_path) as f:
            dev_raw_data = json.load(f)
            time.sleep(3)
        self.log.info ('Tokenizing raw dev data...')
        dev_data_id_list = self.tokenize_raw_data(dev_raw_data)
        self.dev_data_list = self.flatten_data(dev_data_id_list)

        test_json_path = data_path_prefix + '/multiwoz-fine-processed-test.json'
        if self.debugging : test_json_path = data_path_prefix + '/multiwoz-fine-processed-test.json' 
        
        with open(test_json_path) as f:
            test_raw_data = json.load(f)
            time.sleep(3)
        self.log.info ('Tokenizing raw test data...')
        test_data_id_list = self.tokenize_raw_data(test_raw_data)
        self.test_data_list = self.flatten_data(test_data_id_list)

        self.log.info ('The size of raw train, dev ,test%d, %d, and %d' % \
            (len(train_raw_data), len(dev_raw_data), len(test_raw_data)))

        self.dev_num, self.test_num= len(self.dev_data_list), len(self.test_data_list)
        if data_mode == 'train':
            self.log.info ('train turn number is %d, dev turn number is %d, test turn number is %d ' % \
                (len(self.train_data_list), len(self.dev_data_list), len(self.test_data_list)))
            self.shuffle_mode = shuffle_mode
            self.ordering_train_data()
        else:
            pass


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

    def tokenize_raw_data(self, raw_data_list, replace_to_labeled_answer = False): # TODO also get labeld data list and answer
        data_num = len(raw_data_list)
        if self.use_progress:
        all_session_list = []
        for idx in range(data_num):
            one_sess_list = []
            for turn in raw_data_list[idx]: # TODO : also should be labeled list
                one_turn_dict = {}
                for key in turn:
                    if key in ['dial_id', 'pointer', 'turn_domain', 'turn_num', 'aspn', 'dspn', 'aspn_reform', 'db']:
                        one_turn_dict[key] = turn[key]
                    else: # TODO in case of original data, use same, if not, use labeld dataset file
                        # only tokenize ["user", "usdx", "resp", "bspn", "bsdx", "bspn_reform", "bsdx_reform"]
                        value_text = turn[key]
                        if key == 'bspn' and replace_to_labeled_answer == True :
                            dial_turn_idx = '[d]'+turn['dial_id'] + '[t]' + str(turn['turn_num'])
                            try:
                                value_text = self.labeled_data[dial_turn_idx]
                            except:
                                value_text = ''
                        value_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_text))
                        value_id = self.replace_sos_eos_token_id(value_id)
                        one_turn_dict[key] = value_id

                one_sess_list.append(one_turn_dict)
            all_session_list.append(one_sess_list)
        if self.use_progress: p.finish()
        assert len(all_session_list) == len(raw_data_list)
        return all_session_list

    def shuffle_train_data(self):
        random.shuffle(self.train_data_list)

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
        special_tokens = ontology.sos_eos_tokens
        # self.log.info (special_tokens)
        #self.log.info (special_tokens)
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
                # construct belief state data
                # -900 menas get from last character - 900
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
        
    def build_iterator(self, batch_size, mode ):
        batch_list, idx_list = self.get_batches(batch_size, mode)
        for batch, idx in zip(batch_list, idx_list):
            yield batch, idx
            
    def get_batches(self, batch_size, mode): # 여기서 전부다 합친다음 배치사이즈만큼 미리 쪼갠다!
        if mode == 'train':
            batch_list = []
            idx_list = []
            all_data_list = self.train_data_list
        elif mode == 'valid':
            batch_list = []
            idx_list = []
            all_data_list = self.train_data_list
        elif mode == 'test':
            batch_list = []
            idx_list = []
            all_data_list = self.train_data_list

        all_input_data_list, all_output_data_list, all_index_list = [], [], []
        for item in all_data_list:
            dial_turn_key = '[d]'+item['dial_id'] + '[t]' + str(item['turn_num'])
            if dial_turn_key not in self.labeled_data.keys() and mode == 'train_loop':
                continue
            if dial_turn_key in self.labeled_data.keys() and mode == 'tagging'\
                and not self.tagging_all:
                continue

            one_input_data_list = []
            for key in ['bs_input']:
                one_input_data_list.append(item[key])
            all_input_data_list.extend(one_input_data_list)

            one_output_data_list = []
            for key in ['bs_output']:
                one_output_data_list.append(item[key])
            all_output_data_list.extend(one_output_data_list)
            all_index_list.append(dial_turn_key)
        
        data_num = len(all_input_data_list)
        batch_num = int(data_num/batch_size) + 1
        self.train_num = data_num
        

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

        
        return batch_list, idx_list

            


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

