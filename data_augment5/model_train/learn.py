import os
import sys, pdb
import json
import torch
import random
import argparse
import torch.nn as nn
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
import logging
import logging.handlers
from data_augment5.model_train.trainer import  train, evaluate
from data_augment5.model_train.generate_dataclass import Generate_dataclass

log = logging.getLogger('my_log')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')

def init_experiment(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--model_name', type=str, help='t5-base or t5-large')
    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    parser.add_argument('--init_label_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--epoch_num", default=60, type=int, help="Total number of training epochs to perform.")
    
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--eval_batch_size_per_gpu", type=int, default=8, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--debugging", type=int, default=0, help="debugging going small")
    parser.add_argument("--log_interval", type=int, default=1000, help="mini epoch")
    
    return parser.parse_args()


def save_model(model,ckpt_save_path):
    if not os.path.exists(ckpt_save_path):
        os.mkdir(ckpt_save_path)
    # save model
    model.save_pretrained(ckpt_save_path)
    # save tokenizer
    tokenizer.save_pretrained(ckpt_save_path)
        
        
def save_result(epoch, model, one_dev_str):
    log.info ('Saving Model...')
    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str

    if os.path.exists(model_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(model_save_path, exist_ok=True)

    if cuda_available and torch.cuda.device_count() > 1:
        save_model(model.module, model_save_path)
    else:
        save_model(model, model_save_path)

    fileData = {}
    test_output_dir = args.ckpt_save_path
    for fname in os.listdir(test_output_dir):
        if fname.startswith(f'epoch'):
            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime

    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
    max_save_num = 1
    if len(sortedFiles) < max_save_num:
        pass
    else:
        delete = len(sortedFiles) - max_save_num
        for x in range(0, delete):
            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
            log.info (one_folder_name)
            os.system('rm -r ' + one_folder_name)
            
def log_setting(ckpt_save_path, name = None):
    if not name:
        name = "generator"
        
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')
    
    fileHandler = logging.FileHandler(f'{ckpt_save_path}log.txt')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    return log

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise

if __name__ == '__main__':
    special_tokens = ['<sos_b>', '<eos_b>','<sos_u>', '<eos_u>','<context>', '<prev_bspn>', '<this_bspn>']
    
    if torch.cuda.is_available():
        log.info ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            log.info ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            log.info ('Using single GPU training.')
    else:
        pass
 
    args = parse_config()
    device = torch.device('cuda')
    makedirs(args.ckpt_save_path)
    
    log = log_setting(args.ckpt_save_path)
    log.info('seed setting')
    log.info(args)
    init_experiment(args)

    log.info (f'Loading from internet {args.model_name}')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
    model = model.to(device)
    
    log.info('Initialize dataclass')
    
    data = Generate_dataclass(args.model_name, tokenizer, args.data_path_prefix,  args.ckpt_save_path, init_label_path = args.init_label_path, \
        log_path = f'{args.ckpt_save_path}log.txt', debugging = args.debugging)
    
    optimizer = Adafactor(model.parameters(),lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)    
    
    min_loss = 1e10
    max_dev_score, max_dev_str = 0., ''
    
    for epoch in range(args.epoch_num):
        log.info(f'------------------------------Epoch {epoch}--------------------------------------')
        train_loss = train(args,model,optimizer,data,log, cuda_available, device)
        dev_loss = evaluate(args,model,data,log, cuda_available, device)
        
        if dev_loss <  min_loss:
            min_loss = dev_loss
            file_name = 'dev_joint_accuracy_{}'.format(round(dev_loss,2))
            save_result(epoch, model, file_name)
        log.info ('In the  epoch {}, current loss {}, smallest loss {}'.format(epoch,round(dev_loss, 2), round(min_loss, 2)))

