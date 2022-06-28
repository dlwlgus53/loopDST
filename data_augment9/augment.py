# generate text, half and half change or not
import torch
import os, sys
import json
import pdb
import argparse
import copy
import logging
import logging.handlers
import random
import copy
import torch.nn as nn
from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor, T5Config

sys.path.append("../")
from data_augment8.augment import get_generated_dict as get_generated_dict_label_change
from data_augment5.augment import get_generated_dict as get_generated_dict_label_keep
    

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
    parser.add_argument('--aug_num', type=int ,default = 5, help='how many samples will make')
    parser.add_argument('--batch_size', type=int ,default = 10, help='batch_size for t5')
    parser.add_argument('--model_path', type=str ,default = 't5-base', help='batch_size for t5')
    parser.add_argument('--save_path', type=str ,default = './save', help='batch_size for t5')
    parser.add_argument('--number_of_gpu', type=int ,default = 2, help='batch_size for t5')
    
    
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


def split_by_dial(raw_set):
    train_set = []
    test_set = []
    
    total_len = len(raw_set)
    train_len = int(total_len * 0.9)
    
    for idx, dial in enumerate(raw_set):
        if idx < train_len:train_set.append(dial)
        else: test_set.append(dial)
    return train_set, test_set


## This is the most important!
def get_generated_dict(raw_data, tokenizer, model, aug_num, device ,log, log_interval = None):
    number_of_gpu = 1
    batch_size_per_gpu = 10
    
    diff_n = aug_num//2
    same_n = aug_num-diff_n
    
    # 이건 다르겠지
    gen1 = get_generated_dict_label_change(raw_data, tokenizer, model,diff_n,  device, log, log_interval)
    gen2 = get_generated_dict_label_keep(raw_data, tokenizer, model, same_n, device, log, log_interval)
    for key,value in gen2.items():
        aug_idx = int(key.split("[a]")[1]) + diff_n
        new_key = key.split("[a]")[0] + "[a]" + str(aug_idx)
        gen1[new_key] = value
    return gen1
    


def load_model(model_path, device, multi_gpu_training):
    t5_config = T5Config.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
    # if multi_gpu_training:
    #     model = nn.DataParallel(model) # multi-gpu training
    # else:
    #     pass
    model = model.to(device)
    return model



import argparse
if __name__ == '__main__':
    log = log_setting()
    args = parse_config()

    seed_setting(args.seed)
   
    if torch.cuda.device_count() > 1:
        multi_gpu_training = True
    else:
        multi_gpu_training = False
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
    model = load_model(args.model_path, device, multi_gpu_training)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    raw_datapath = args.data_path_prefix + 'multiwoz-fine-processed-tenpercent.json' # 전체 training data

    with open(raw_datapath) as f:
        raw_data = json.load(f)
    # get_generated_dict(raw_data, tokenizer, model, topn,  number_of_gpu, batch_size_per_gpu, device, log, log_interval = None):
    generated_dict =get_generated_dict(raw_data, tokenizer, model, args.aug_num,\
        device, log, log_interval = None)
    
    raw_data_similar = []
    for dial_idx, dial in enumerate(raw_data):
        # if dial_idx%30 == 0 and dial_idx !=0:log.info(f'saving dials {dial_idx}/{len(sraw_data)} done')
        for n in range(args.aug_num):
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
    makedirs(f"./{args.save_path}")


    train_set, test_set = split_by_dial(raw_data_similar)
    # 90%
    with open(f'./{args.save_path}/multiwoz-fine-processed-train.json', 'w') as outfile:
        json.dump(train_set, outfile, indent=4)
        
    # 10%
    with open(f'./{args.save_path}/multiwoz-fine-processed-dev.json', 'w') as outfile:
        json.dump(test_set, outfile, indent=4)
        