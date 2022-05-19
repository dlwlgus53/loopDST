# https://jimmy-ai.tistory.com/196
# augment data 저장
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
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from collections import defaultdict
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
    parser.add_argument('--topn', type=int ,default = 5, help='how many samples will make')
    parser.add_argument('--batch_size', type=int ,default = 10, help='batch_size for t5')
    parser.add_argument('--model_path', type=str ,default = 't5-base', help='batch_size for t5')
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


def mask_to_text(tokenizer, mask_arr, tokenized_input):
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
    return new_text



def get_mask_arr(overlap_position, input_ids, start_token, end_token):
    mask_arr = torch.zeros(input_ids.shape)
    for p in range(len(mask_arr)):
        random_num = random.random()
        if p not in overlap_position and random_num<0.1:
            mask_arr[p] = True
    mask_arr = mask_arr * (input_ids!= start_token) * (input_ids != end_token) # 시작/끝 토큰도 바꾸지 않는다.
    return mask_arr



# def change(tokenizer, model, input_text, label):
#     input_text = remove_special_character(input_text)
#     label = remove_special_character(label)
#     tokenized_input = tokenizer(input_text, return_tensors='pt')
#     input_ids = tokenized_input.input_ids.detach().clone()[0]
#     label_ids = tokenizer(label, return_tensors='pt').input_ids.detach().clone()[0]
#     overlap_position = get_label_position(input_ids, label_ids) # DST label과 겹치는 token의 위치
#     start_token = 1
#     end_token =2

#     if len(label.strip()) ==0 or len(overlap_position) == 0: # label의 결과가 없거나, label과 input text가 겹치는게 없는 경우 이 대화는 사용하지 않는다.
#         new_text = '-1'
#     else:
#         mask_arr =  get_mask_arr(overlap_position, input_ids, start_token, end_token))    
#         new_text = mask_to_text(tokenizer, mask_arr, tokenized_input)
#         new_text.replace("centre","center")

#         print("wrong option!")

#     if new_text == input_text and option == 'different': # 마스크 추론 결과 기존의 text와 동일하다면, 그리고 option이 different라면 이 케이스는 쓰지 않는다.
#         new_text = '-1'
        
#     masked = tokenizer.decode(tokenized_input.input_ids[0])
#     new_text = '<sos_u> ' + new_text + ' <eos_u>'
#     return masked, new_text

    
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
    
    
def tokenize(input_text,label, tokenizer):
    input_text = remove_special_character(input_text)
    label = remove_special_character(label)
    tokenized_input = tokenizer(input_text, return_tensors='pt')
    input_ids = tokenized_input.input_ids.detach().clone()[0]
    label_ids = tokenizer(label, return_tensors='pt').input_ids.detach().clone()[0]
    overlap_position = get_overlap_position(input_ids, label_ids) # DST label과 겹치는 token의 위치
    start_token = 2
    end_token =1
    
    mask_arr =  get_mask_arr(overlap_position, input_ids, start_token, end_token)
    selection = torch.flatten((mask_arr).nonzero()).tolist()
    
    for idx, mask_position in enumerate(selection):
        mask = f'<extra_id_{idx}>'
        mask_idx = tokenizer.encode(mask)[0]
        tokenized_input.input_ids[0,mask_position] = mask_idx 
    return tokenized_input.input_ids[0].tolist()

def get_will_change_item(raw_data,init_labeled_data, tokenizer):
    tokenized_masked_list = []
    dial_turn_id_list = []
    for dial_idx, dial in enumerate(raw_data):
        if dial_idx%30 == 0 and dial_idx !=0:
            log.info(f'{dial_idx}/{len(raw_data)}')
        dial_turn_key = '[d]'+ dial[0]['dial_id'] + '[t]' + '0'
        if dial_turn_key not in init_labeled_data: continue
        for turn in dial:
            for n in range(args.topn):
                dial_turn_key = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num'])
                text = turn['user']
                label = turn['bspn']
                tokenize_mask_text = tokenize(text, label,tokenizer)
                dial_turn_id_list.append(dial_turn_key)
                tokenized_masked_list.append(tokenize_mask_text)
    return dial_turn_id_list, tokenized_masked_list


def generate_new_text(model, dial_turn_id_list, tokenized_masked_list, batch_size, DEVICE):
    start = 0    
    generated_dict = {}
    count_dict = defaultdict(int) # default dict
    pdb.set_trace()
    while True:
        batch_id  = dial_turn_id_list[start:start+batch_size]
        batch  = tokenized_masked_list[start:start+batch_size]
        batch = tokenizer.pad({'input_ids' :batch})
        batch = torch.tensor(batch.input_ids).to(DEVICE)
        generated = model.generate(batch)
        pdb.set_trace()
        for output in generated:
            decoded = tokenizer.decode(output)
            pdb.set_trace()
        # do something
        generated_text = []
        
        for idx, text in zip(batch_id, generated_text):
            count_dict[idx] +=1
            dial_turn_count_id = idx +'[a]' + str(count_dict[idx])
            generated_dict[dial_turn_count_id] = text
            
        start += batch_size
        
        if start> len(dial_turn_id_list):
            break
    return generated_dict


import argparse
if __name__ == '__main__':
    log = log_setting()
    args = parse_config()

    log.info('seed setting')
    seed_setting(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    config = T5Config.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path, config=config).to(DEVICE)

    raw_datapath = args.data_path_prefix + 'multiwoz-fine-processed-train.json' # 전체 training data
    raw_init_datapath = args.data_path_prefix + 'labeled_init.json' # 10% 사용할 때, 어떤 10%를 사용할 지 정보를 가지고 있는 파일
    
    with open(raw_init_datapath) as f:
        init_labeled_data = json.load(f)

    with open(raw_datapath) as f:
        raw_data = json.load(f)
    
    dial_turn_id_list, tokenized_masked_list = get_will_change_item(raw_data,init_labeled_data, tokenizer)
    generated_dict= generate_new_text(model, dial_turn_id_list, tokenized_masked_list, args.batch_size, DEVICE)
    
    
    
    raw_data_similar = []
    for dial_idx, dial in enumerate(raw_data):
        if dial_idx%30 == 0 and dial_idx !=0:log.info(f'{dial_idx}/{len(raw_data)}')
        dial_turn_key = '[d]'+ dial[0]['dial_id'] + '[t]' + '0'
        if dial_turn_key not in init_labeled_data: continue
            
        for n in range(args.topn+1):
            if n==0:
                similar_dial = copy.deepcopy(dial)
            else:
                for turn in dial:
                    similar_dial = []
                    for turn in dial:
                        idx = dial_turn_key + '[c]' + str(n)
                        similar_turn = copy.deepcopy(turn)
                        if idx in generated_dict:
                            similar_turn['user'] = generated_dict[idx]
                        similar_dial.append(similar_turn)

            raw_data_similar.append(similar_dial)

    makedirs(f"./{args.save_path}")

    with open(f'./{args.save_path}/similar.json', 'w') as outfile:
        json.dump(raw_data_similar, outfile, indent=4)
        