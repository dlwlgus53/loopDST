
import torch
import os
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
from .model_train.generate_dataclass import Generate_dataclass


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

def remove_special_character(text):
    for token in sos_eos_tokens:
        text = text.replace(token, '')
    text = text.replace( '[', '')
    text = text.replace( ']', '')
    return text.strip()

def get_overlap_position(input_ids, label_ids):
    input_ids = input_ids.tolist()
    label_ids = label_ids.tolist()
    
    # Have to see this again
    
    overlap_index = []
    for label_value in label_ids:
        if label_value == 0 or label_value == 2:
            continue
        overlap_index += [i for i, x in enumerate(input_ids) if x == label_value]
    return overlap_index
def get_mask_arr(overlap_position, input_ids, start_token, end_token, change_rate):
    mask_arr = torch.zeros(input_ids.shape)
    for p in range(len(mask_arr)):
        random_num = random.random()
        if p not in overlap_position and random_num<change_rate:
            mask_arr[p] = True
    mask_arr = mask_arr * (input_ids!= start_token) * (input_ids != end_token) # 시작/끝 토큰도 바꾸지 않는다.
    return mask_arr


    
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
    
    
def tokenize(input_text,label, tokenizer, change_rate):
    input_text = remove_special_character(input_text)
    label = remove_special_character(label)
    tokenized_input = tokenizer(input_text, return_tensors='pt')
    input_ids = tokenized_input.input_ids.detach().clone()[0]
    label_ids = tokenizer(label, return_tensors='pt').input_ids.detach().clone()[0]
    overlap_position = get_overlap_position(input_ids, label_ids) # DST label과 겹치는 token의 위치
    start_token = 0
    end_token =2
    mask_idx = 50264
    
    mask_arr =  get_mask_arr(overlap_position, input_ids, start_token, end_token, change_rate)
    selection = torch.flatten((mask_arr).nonzero()).tolist()
    
    for idx, mask_position in enumerate(selection):
        tokenized_input.input_ids[0,mask_position] = mask_idx 
    return tokenized_input.input_ids[0].tolist()


def generate_new_text(model, data, device,log, aug_num, number_of_gpu, batch_size_per_gpu, log_interval=None):
    model.eval()
    generate_dict = {}

    gen_iterator = data.build_iterator(batch_size=number_of_gpu * batch_size_per_gpu, mode = "gen", aug_num =aug_num)
    with torch.no_grad():
        for idx, (gen_batch, key, label)in enumerate(gen_iterator):
            if idx == 0:
                dev_num = len(data.gen_data_list)
                dev_batch_num_per_epoch = int(dev_num / (number_of_gpu * batch_size_per_gpu))+1
            idx += 1
            # if idx%50 == 0: log.info(f'{idx*100/dev_batch_num_per_epoch:.2f} %')
            one_dev_input_batch, one_dev_output_batch = gen_batch
            if len(one_dev_input_batch) == 0 or len(one_dev_output_batch) == 0: break
            source_input, _, _, _ = \
            data.parse_batch_tensor(gen_batch)
            input_ids = source_input.to(device)
            outputs = model.generate(input_ids = input_ids)
            for k, output, bspn in zip(key, outputs, label):
                text = data.tokenizer.decode(output,skip_special_tokens = True)
                generate_dict[k] = {'text' : text, 'bspn' : bspn}
    return generate_dict

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
    data = Generate_dataclass(tokenizer, raw_data = raw_data,  log = log, debugging = False)
    generated_dict= generate_new_text(model, data, device, log, number_of_gpu, batch_size_per_gpu, aug_num, log_interval)
    return generated_dict
    


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
        