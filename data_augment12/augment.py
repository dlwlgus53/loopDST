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
import argparse


all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

sos_eos_tokens = ['<_PAD_>', '<go_r>', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<go_d>','<eos_d>', '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', 
                '<sos_db>', '<eos_db>', '<sos_context>', '<eos_context>']



random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']

# Insert punction words into a given sentence with the given ratio "punc_ratio"

def _insert_punctuation_marks(sentence, punc_ratio=0.3):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line


def get_generated_dict(dataset, aug_num, log): 
    generated_dict = {}
    for dial in dataset:
        for turn in dial:
            text = turn['user']
            text = text.replace("<sos_u>" , "").replace("<eos_u>", "").strip()
            for n in range(aug_num):
                dial_turn_key = '[d]'+ turn['dial_id'] + '[t]' + str(turn['turn_num']) + '[a]' + str(n)
                sentence_aug = _insert_punctuation_marks(text)
                dst_sentence = '<sos_u> ' + sentence_aug + ' <eos_u>'
                generated_dict[dial_turn_key] = {'text' : dst_sentence}
    return generated_dict

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
    parser.add_argument('--aug_num', type=int ,default = 5, help='how many samples will make')
    parser.add_argument('--batch_size', type=int ,default = 10, help='batch_size for t5')
    parser.add_argument('--save_path', type=str ,default = './save', help='batch_size for t5')
    return parser.parse_args()

def remove_special_character(text):
    for token in sos_eos_tokens:
        text = text.replace(token, '')
    text = text.replace( '[', '')
    text = text.replace( ']', '')
    return text.strip()

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






if __name__ == '__main__':
    log = log_setting()
    args = parse_config()

    log.info('seed setting')
    seed_setting(args.seed)
    
    raw_datapath = args.data_path_prefix + 'multiwoz-fine-processed-small.json' # 전체 training data
    with open(raw_datapath) as f:
        raw_data = json.load(f)
        
    generated_dict = get_generated_dict(raw_data, args.aug_num, log)
    raw_data_similar = []
    
    ###################### dont' change #########################
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
        