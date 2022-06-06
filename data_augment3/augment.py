# https://jimmy-ai.tistory.com/196
# augment data 저장
# augmen only!
import random
import torch
import os
import json
import pdb
import argparse
import copy
import logging
import logging.handlers
import copy
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

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

def get_will_change_item(raw_data, tokenizer, change_rate, topn, log):
    tokenized_masked_list = []
    dial_turn_id_list = []
    for dial_idx, dial in enumerate(raw_data):
        for turn in dial:
            for n in range(topn):
                dial_turn_key = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                text = turn['user']
                label = turn['bspn']
                tokenize_mask_text = tokenize(text, label,tokenizer, change_rate)
                dial_turn_id_list.append(dial_turn_key)
                tokenized_masked_list.append(tokenize_mask_text)
    return dial_turn_id_list, tokenized_masked_list


def generate_new_text(model, tokenizer, dial_turn_id_list, tokenized_masked_list, batch_size, DEVICE,log, log_interval=None):
    if not log_interval:
        log_interval = 100
    start = 0    
    generated_dict = {}
    count_dict = defaultdict(int) # default dict
    while True:
        if start %log_interval ==0:
            log.info(f"generate new text {start}/{len(dial_turn_id_list)} done")
        batch_id  = dial_turn_id_list[start:start+batch_size]
        batch  = tokenized_masked_list[start:start+batch_size]
        batch = tokenizer.pad({'input_ids' :batch})
        batch = torch.tensor(batch.input_ids).to(DEVICE)
        generated = model(batch)
        generated = torch.argmax(generated.logits, dim = -1)

        
        for idx, masked, output in zip(batch_id, batch, generated):
            end_position = (masked==2).nonzero().item()
            decode_result = tokenizer.decode(output[1:end_position])
            mask_text = tokenizer.decode(masked[1:end_position])
            decode_result = "<sos_u> "+ decode_result.replace("</s>","") + " <eos_u>"
            count_dict[idx] +=1
            generated_dict[idx] = {'text' :decode_result, 'mask_text' : mask_text}
            
        start += batch_size
        
        if start>= len(dial_turn_id_list):
            break
    return generated_dict

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

def get_generated_dict(raw_data, tokenizer, model, change_rate, topn,  batch_size, device, log, log_interval):
    dial_turn_id_list, tokenized_masked_list = get_will_change_item(raw_data, tokenizer, change_rate, topn,log)
    generated_dict= generate_new_text(model, tokenizer, dial_turn_id_list, tokenized_masked_list, batch_size, device, log, log_interval)
    return generated_dict

import argparse
if __name__ == '__main__':
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
    dial_turn_id_list, tokenized_masked_list = get_will_change_item(raw_data, tokenizer, args.change_rate, args.topn,log)
    generated_dict= generate_new_text(model, tokenizer, dial_turn_id_list, tokenized_masked_list, args.batch_size, DEVICE, log)
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
        