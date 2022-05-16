# https://jimmy-ai.tistory.com/196

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
from transformers import pipeline


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


def change(tokenizer, model, input_text, label, model_special_tokens, option):
    new_text = ''
    input_text = remove_special_character(input_text)
    label = remove_special_character(label)
    
    tokenized_input = tokenizer(input_text, return_tensors='pt')
    input_ids = tokenized_input.input_ids.detach().clone()[0]
    label_ids = tokenizer(label, return_tensors='pt').input_ids.detach().clone()[0]
    overlap_position = get_label_position(input_ids, label_ids) # DST label과 겹치는 token의 위치
    mask_arr = torch.zeros(input_ids.shape)


    if option == 'similar':
        if len(label.strip()) ==0 or len(overlap_position) == 0: # label의 결과가 없거나, label과 input text가 겹치는게 없는 경우 이 대화는 사용하지 않는다.
            new_text = '-1'
        else:
            for p in range(len(mask_arr)):
                if p not in overlap_position and random.random()<0.3:
                    mask_arr[p] = True
            mask_arr = mask_arr * (input_ids!= model_special_tokens['start']) * (input_ids != model_special_tokens['end']) # 시작/끝 토큰도 바꾸지 않는다.
            
            # rand = torch.rand(input_ids.shape) 
            # mask_arr = rand < 0.30 # 전체의 30%를 mask token으로 변경
            # for position in overlap_position: # 그중, label과 겹치는 위치는 mask token으로 바꾸지 않는다.
            #     mask_arr[position] = False
            # mask_arr = mask_arr * (input_ids!= model_special_tokens['start']) * (input_ids != model_special_tokens['end']) # 시작/끝 토큰도 바꾸지 않는다.
            

    elif option == 'different':
        if len(label.strip()) ==0 or len(overlap_position) == 0: # label의 결과가 없거나, label과 input text가 겹치는게 없는 경우 이 대화는 사용하지 않는다.
            new_text = '-1'
        else:
            mask_arr = torch.zeros(input_ids.shape)
            flag = False
            for position in overlap_position: # label과 겹치는 위치 중
                # if random.random()<0.5 : # 50%를 mask token으로 변경
                #     flag = True
                mask_arr[position] = True
            # if flag == False: # 우연히 하나도 바뀌지 않은 경우
            #     mask_arr[random.choice(overlap_position)] = True # 랜덤하게 하나를 골라서 mask token으로 바꿔준다.
    else:
        print("wrong option!")

    if new_text != '-1':
        selection = torch.flatten((mask_arr).nonzero())
        tokenized_input.input_ids[0,selection] = model_special_tokens['mask'] # true인 부분을 mask token으로 바꿔주고
        outputs = model(**tokenized_input) # 모델에 넣어서
        prediction = torch.argmax(outputs.logits, dim = -1) # 결과를 얻는다.
        new_text = tokenizer.decode(prediction[0]) # masked
        new_text = new_text.replace('<s>','')
        new_text = new_text.replace('</s>','')
        new_text = new_text.replace('\'',' \'')
        new_text = new_text.replace('.',' .')
        new_text = new_text.replace('?',' ?')
        new_text = new_text.replace('!',' !')
        new_text = new_text.lower()

    if new_text == input_text and option == 'different': # 마스크 추론 결과 기존의 text와 동일하다면, 그리고 option이 different라면 이 케이스는 쓰지 않는다.
        new_text = '-1'
        
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
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

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
            # if dial_idx == 90:
            #     breaks
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
            
            
            changed_text_similar = change(tokenizer, model, input_text, label, model_special_tokens, 'similar')
            changed_text_different = change(tokenizer, model, input_text, label, model_special_tokens, 'different')
            
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