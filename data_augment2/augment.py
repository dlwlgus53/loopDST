# https://jimmy-ai.tistory.com/196
import sys
sys.path.append('../')
import random
import torch
import os
import json
import pdb
import argparse
import copy
import logging
import numpy as np
import logging.handlers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from transformers import T5Tokenizer
from DST.modelling.T5Model import T5Gen_Model



all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

sos_eos_tokens = ['<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<go_d>','<eos_d>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', 
                '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>']

def seed_setting(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--seed', type=int ,default = 1, help='The path where the data stores.')
    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    
    return parser.parse_args()


def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise

def remove_special_character(text):
    for token in sos_eos_tokens:
        text = text.replace(token, '')
    text = text.replace( '[', '')
    text = text.replace( ']', '')
    return text.strip()

def get_label_position(input_ids, label_ids):
    input_ids = input_ids.tolist()
    label_ids = label_ids.tolist()

    overlap_index = []
    for label_value in label_ids:
        if label_value == 0 or label_value == 2:
            continue
        overlap_index += [i for i, x in enumerate(input_ids) if x == label_value]
    return overlap_index


def change(input_text, option):
    
    # input text의 attention을 구해서
    # option == 'similar' 라면
        # attention 값이 크지 않은 단어를 찾고
        # 그 단어를 <mask> 로 바꾼 뒤
    # option == 'different'라면
        # attention이 큰 단어를 찾고
        # 그 단어를  <mask>로 바꾼 뒤

    # roberta로 mask filling

    # 이 구조로 저장해야함.!
    new_text = '<sos_u> ' + new_text + ' <eos_u>'
    return new_text

    
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise    
    
def log_setting():
    log = logging.getLogger('my_log')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')
    
    fileHandler = logging.FileHandler(f'log.txt')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    return log
    

def load_tokenizer(pretrained_path):
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
    special_tokens = sos_eos_tokens
    tokenizer.add_tokens(special_tokens)
    return tokenizer


def load_pretrained_model(pretrained_path, tokenizer):
    add_special_decoder_token = True
    cuda_available = torch.cuda.is_available()
    dropout = 0.1
    model = T5Gen_Model(pretrained_path, tokenizer, sos_eos_tokens, dropout=dropout, 
        add_special_decoder_token=add_special_decoder_token, is_training=True)

    if cuda_available:
        device = torch.device('cuda')
        model = model.to(device)
    else:
        pass
    return model



import argparse
if __name__ == '__main__':
    log = log_setting()
    args = parse_config()
    model_configuration = RobertaConfig()
    model_special_tokens = {
        'mask' : 50264,
        'start' : 0,
        'end' : 2
    }
    log.info('seed setting')
    seed_setting(args.seed)

    # Roberta is for mask filling.
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    
    # T5 model is for get attention.
    log.info("tokenizer and model load")
    tokenizer_for_attention = load_tokenizer(args.pretrained_path)
    model_for_attention =load_pretrained_model(args.pretrained_path, tokenizer_for_attention)
    
    model.eval()
    model_for_attention.eval()

    raw_datapath = args.data_path_prefix + 'multiwoz-fine-processed-small.json' # 전체 training data
    raw_init_datapath = args.data_path_prefix + 'labeled_init.json' # 10% 사용할 때, 어떤 10%를 사용할 지 정보를 가지고 있는 파일
    
    with open(raw_init_datapath) as f:
        init_labeled_data = json.load(f)

    with open(raw_datapath) as f:
        raw_data = json.load(f)
    
    raw_data_similar = []
    raw_data_different = []
    
    for dial_idx, dial in enumerate(raw_data):
        if dial_idx%30 == 0 and dial_idx !=0:
            print(f'{dial_idx}/{len(raw_data)}')
            break
        similar_dial = []
        different_dial = []
        for turn in dial:
            dial_turn_key = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num'])
            if dial_turn_key not in init_labeled_data: # 10% 만 사용하기 위한 if문
                continue

            # 실제 DST인풋으로 들어가는건  turn에서 user 뿐 그래서 user만 사용
            similar_turn = {
                'user' : turn['user'],
                'bspn' : turn['bspn']
            }
            different_turn = {
                'user' : turn['user'],
                'bspn' : turn['bspn']
            }
            # similar_turn = copy.deepcopy(turn)
            # different_turn = copy.deepcopy(turn)
            
            input_text = turn['user']
            label = turn['bspn']
            
            # TODO : change 함수 만들기!
            changed_text_similar = change(input_text, 'similar')
            changed_text_different = change(input_text, 'different')
            
            if changed_text_similar:
                similar_turn['user_similar'] = changed_text_similar
            if changed_text_different:
                different_turn['user_different'] = changed_text_different
                
            similar_dial.append(similar_turn)
            different_dial.append(different_turn)
        raw_data_similar.append(similar_dial)
        raw_data_different.append(different_dial)

    
    makedirs("./save")
    with open('save/similar.json', 'w') as outfile:
        json.dump(raw_data_similar, outfile, indent=4)

    with open('save/different.json', 'w') as outfile:
        json.dump(raw_data_different, outfile, indent=4)