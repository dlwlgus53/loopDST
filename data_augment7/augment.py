# https://jimmy-ai.tistory.com/196
# augment data 저장
# augmen only!
import random
import torch
import sys
import os
import json
import pdb
import argparse
import copy
import logging
import logging.handlers
import copy
from collections import defaultdict
sys.path.append("../")
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from data_augment2.augment import get_generated_dict as get_generated_dict_label_change
from data_augment3.augment import get_generated_dict as get_generated_dict_label_keep
    
# data_augment2
# data_augment
    

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
    parser.add_argument('--init_label_path', type=str, help='The path where the data stores.')
    parser.add_argument('--seed', type=int ,default = 1, help='The path where the data stores.')
    parser.add_argument('--topn', type=int ,default = 5, help='how many samples will make')
    parser.add_argument('--batch_size', type=int ,default = 10, help='batch_size for t5')
    parser.add_argument('--model_path', type=str ,default = 't5-base', help='batch_size for t5')
    parser.add_argument('--change_rate', type=float ,default = 0.3, help='batch_size for t5')
    parser.add_argument('--save_path', type=str ,default = './save', help='batch_size for t5')
    
    return parser.parse_args()

    
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise    
    
def log_setting(name = None):
    if not name :
        name = "aug"
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')
    
    fileHandler = logging.FileHandler(f'log.txt')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    return log
    

def filtering_data(raw_data, filter_data):
    data = []
    for dial in raw_data:
        dial_turn_key = '[d]'+ dial[0]['dial_id'] + '[t]0'
        if dial_turn_key in filter_data:
            data.append(dial)
    return data


def split_by_dial(raw_set):
    train_set = []
    test_set = []
    
    total_len = len(raw_set)
    train_len = int(total_len * 0.9)
    
    for idx, dial in enumerate(raw_set):
        if idx < train_len:train_set.append(dial)
        else: test_set.append(dial)
    return train_set, test_set

# input shuold not be change
def get_generated_dict(raw_data, tokenizer, model, change_rate, topn,  batch_size, device, log, log_interval = None):
    diff_n = topn//2
    same_n = topn-diff_n
    gen1 = get_generated_dict_label_change(raw_data, tokenizer, model, change_rate, diff_n,  batch_size, device, log, log_interval)
    gen2 = get_generated_dict_label_keep(raw_data, tokenizer, model, change_rate, same_n,  batch_size, device, log, log_interval)
    for key,value in gen2.items():
        aug_idx = int(key.split("[a]")[1]) + diff_n
        new_key = key.split("[a]")[0] + "[a]" + str(aug_idx)
        gen1[new_key] = value
    return gen1

import argparse
if __name__ == '__main__':
    print("works")
    log = log_setting()
    args = parse_config()

    log.info('seed setting')
    seed_setting(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig()
    model = RobertaForMaskedLM.from_pretrained('roberta-base').to(DEVICE)

    raw_datapath = args.data_path_prefix + 'multiwoz-fine-processed-train.json' # 전체 training data
    raw_init_datapath = args.init_label_path + 'labeled_init.json' # 10% 사용할 때, 어떤 10%를 사용할 지 정보를 가지고 있는 파일
    
    with open(raw_init_datapath) as f:
        init_labeled_data = json.load(f)

    with open(raw_datapath) as f:
        raw_data = json.load(f)
    raw_data = filtering_data(raw_data, init_labeled_data)
    generated_dict = get_generated_dict(raw_data, tokenizer, model, args.change_rate, args.topn,  args.batch_size, 'cuda', log)
    raw_data_similar = []
    for dial_idx, dial in enumerate(raw_data):
        if dial_idx%30 == 0 and dial_idx !=0:log.info(f'saving dials {dial_idx}/{len(raw_data)} done')
        for n in range(args.topn):
            similar_dial = []
            for turn in dial:
                idx = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                similar_turn = copy.deepcopy(turn)
                similar_turn['dial_id'] += f'_v{str(n)}'
                similar_turn['user'] = generated_dict[idx]['text']
                similar_turn['mask'] = generated_dict[idx]['mask_text']
                if 'label' in  generated_dict[idx]:
                    similar_turn['bspn'] =  generated_dict[idx]['label']
                if 'mask_label' in  generated_dict[idx]:
                    similar_turn['mask_bspn'] = generated_dict[idx]['mask_label']
                similar_dial.append(similar_turn)
            raw_data_similar.append(similar_dial)
    makedirs(f"./{args.save_path}")
    train_set, test_set = split_by_dial(raw_data_similar)
    # 90%
    with open(f'./{args.save_path}/multiwoz-fine-processed-train.json', 'w') as outfile:
        json.dump(train_set, outfile, indent=4)
        
    # 10%
    with open(f'./{args.save_path}/multiwoz-fine-processed-dev.json', 'w') as outfile:
        json.dump(test_set, outfile, indent=4)
        