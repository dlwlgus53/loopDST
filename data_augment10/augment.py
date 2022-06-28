# back translation
# https://towardsdatascience.com/data-augmentation-in-nlp-using-back-translation-with-marianmt-a8939dfea50a

# 왜 결과물이 똑같지?
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
from transformers import MarianMTModel, MarianTokenizer






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
    
    


def perform_translation(batch_texts, model, tokenizer, language="fr"):
    formated_batch_texts = format_batch_texts(language, batch_texts)
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

def format_batch_texts(language_code, batch_texts):
  
  formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

  return formated_bach

def generate_new_text(data, model, tokenizer, batch_size, lan_code, aug_num = None, log_interval = None):
    if not log_interval:
        log_interval = 100
        
    start = 0    
    generated_dict = {}
    model = model.to("cuda")
    keys = list(data.keys())
    iter =0 
    while True:
        iter+= 1
        if iter % 30 == 0:
            print(f"progress {iter*100/(len(data)/batch_size):.2f}")
        key_batch  = keys[start:start+batch_size]
        text_batch  = [data[k] for k in key_batch]
        
        formated_batch_texts = format_batch_texts(lan_code, text_batch)
        tokenized = tokenizer(formated_batch_texts, return_tensors="pt", padding=True)
        for key, _ in tokenized.items():
            tokenized[key] = tokenized[key].to('cuda')
        if aug_num:
            translated = model.generate(**tokenized,  num_return_sequences =aug_num ,num_beams = aug_num + 2)
            translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            key_batch_reform = [item + "[a]" + str(i) for item in key_batch for i in range(aug_num)]
        else:
            translated = model.generate(**tokenized)
            translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            key_batch_reform = key_batch
            
        for key, output in zip(key_batch_reform, translated_texts):
            generated_dict[key] = output
        start += batch_size
        if start>= len(keys):
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

# 이거 나중에 여기서 하도록 바꿔야 겠네

def format_batch_texts(language_code, batch_texts):
  formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
#   no_bana = { key:val for key,val in fruit_color.items() if key != "banana" }
  return formated_bach

def data_processing(raw_data): # TODO
    data = {}
    for dial in raw_data:
        for turn in dial:
            key = '[d]'+turn['dial_id'] + '[t]' + str(turn['turn_num'])
            data[key] = turn['user'].replace("<sos_u> ","").replace(" <eos_u>","")
    return data

def get_generated_dict(raw_data, tokenizer1, tokenizer2,  model1, model2, aug_num,  batch_size, device, log, log_interval= None):
    org_data= data_processing(raw_data) # key:value dictionary
    fr_data= generate_new_text(org_data, model1, tokenizer1, batch_size, aug_num = aug_num, lan_code='fr')
    en_data= generate_new_text(fr_data, model2, tokenizer2, batch_size, lan_code='en' )
    final_data = {}
    for key in en_data:
        final_data[key] = {'text' : "<sos_u> " + en_data[key] + " <eos_u>"}
    return final_data

import argparse
if __name__ == '__main__':
    log = log_setting()
    args = parse_config()

    log.info('seed setting')
    seed_setting(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
    
    model_name1 = 'Helsinki-NLP/opus-mt-en-fr'
    model_name2 = 'Helsinki-NLP/opus-mt-fr-en'
    
    tokenizer1 = MarianTokenizer.from_pretrained(model_name1)
    model1 = MarianMTModel.from_pretrained(model_name1).to(DEVICE)

    tokenizer2 = MarianTokenizer.from_pretrained(model_name2)
    model2 = MarianMTModel.from_pretrained(model_name2).to(DEVICE)
    
    raw_datapath = args.data_path_prefix + 'multiwoz-fine-processed-train.json' # 전체 training data
    raw_init_datapath = args.init_label_path + 'labeled_init.json' # 10% 사용할 때, 어떤 10%를 사용할 지 정보를 가지고 있는 파일
    
    with open(raw_init_datapath) as f:
        init_labeled_data = json.load(f)

    with open(raw_datapath) as f:
        raw_data = json.load(f)
        
    raw_data = filtering_data(raw_data, init_labeled_data)
    raw_data = raw_data[:50]
    
    generated_dict = get_generated_dict(raw_data = raw_data, tokenizer1 = tokenizer1,
                                        tokenizer2 = tokenizer2,  model1 = model1,
                                        model2 = model2, aug_num = args.aug_num,
                                        batch_size = args.batch_size, device = 'cuda', 
                                        log = log)
    
    
    raw_data_similar = []
    for dial_idx, dial in enumerate(raw_data):
        if dial_idx%30 == 0 and dial_idx !=0:log.info(f'saving dials {dial_idx}/{len(raw_data)} done')
        for n in range(args.aug_num):
            similar_dial = []
            for turn in dial:
                idx = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                similar_turn = copy.deepcopy(turn)
                similar_turn['dial_id'] += f'_v{str(n)}'
                similar_turn['user'] = generated_dict[idx]['text'] 
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
        