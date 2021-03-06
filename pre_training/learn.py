import os
import sys, pdb
sys.path.append('../DST')
from modelling.T5Model import T5Gen_Model
import json
import torch
import random
import argparse
import operator
import torch.nn as nn
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
from transformers import T5Tokenizer
from dataclass import NLIdata

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import logging
import logging.handlers
import copy
from trainer import train, evaluate

log = logging.getLogger('my_log')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')


def init_experiment(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')

    parser.add_argument('--model_name', type=str, help='t5-base or t5-large or facebook/bart-base or facebook/bart-large')

    # training configuration
    parser.add_argument('--optimizer_name', default='adafactor', type=str, help='which optimizer to use during training, adam or adafactor')
    parser.add_argument('--specify_adafactor_lr', type=str, default='True', help='True or False, whether specify adafactor lr')
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
    
    return parser.parse_args()

def get_optimizers(model, args, specify_adafactor_lr):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # overall_batch_size = args.number_of_gpu * args.batch_size_per_gpu * args.gradient_accumulation_steps
    # num_training_steps = train_num * args.epoch_num // overall_batch_size
    if args.optimizer_name == 'adam':
        # log.info ('Use Adam Optimizer for Training.')
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        pass
    elif args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule
        if specify_adafactor_lr:
            log.info ('Specific learning rate.')
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=1e-3,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            scheduler = None
        else:
            log.info ('Do not specific learning rate.')
            optimizer = Adafactor(optimizer_grouped_parameters, 
                scale_parameter=True, 
                relative_step=True, 
                warmup_init=True, 
                lr=None)
            scheduler = AdafactorSchedule(optimizer)
    else:
        raise Exception('Wrong Optimizer Name!!!')
    return optimizer, scheduler

def load_model(args, data, cuda_available, load_pretrained = True):
    add_special_decoder_token = True
    log.info ('Start loading model...')
    model = T5Gen_Model(args.model_name, data.tokenizer, data.special_token_list, dropout=args.dropout, 
        add_special_decoder_token=add_special_decoder_token, is_training=True)
    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    log.info ('Model loaded')
    return model


def load_optimizer(model, args,  specify_adafactor_lr):
    optimizer, scheduler = get_optimizers(model, args, specify_adafactor_lr)
    optimizer.zero_grad()
    return optimizer, scheduler
    

def save_result(epoch, model, one_dev_str,all_dev_result):
    log.info ('Saving Model...')
    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str

    import os
    if os.path.exists(model_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(model_save_path, exist_ok=True)

    if cuda_available and torch.cuda.device_count() > 1:
        model.module.save_model(model_save_path)
    else:
        model.save_model(model_save_path)

    import json
    pkl_save_path = model_save_path + '/' + one_dev_str + '.json'
    with open(pkl_save_path, 'w') as outfile:
        json.dump(all_dev_result, outfile, indent=4)

    import os
    from operator import itemgetter
    fileData = {}
    test_output_dir = args.ckpt_save_path
    for fname in os.listdir(test_output_dir):
        if fname.startswith(f'epoch_{epoch}'):
            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
        else:
            pass
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

def log_setting(args):
    log = logging.getLogger('my_log')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] > %(message)s')
    
    fileHandler = logging.FileHandler(f'{args.ckpt_save_path}/log.txt')
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

import argparse
if __name__ == '__main__':
    args = parse_config()
    makedirs(args.ckpt_save_path)
    log = log_setting(args)
    
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
    

    log.info(args)
    init_experiment(args.seed)
    device = torch.device('cuda')
    
    assert args.model_name.startswith('t5')

    log.info (f'Load tokenizer from internet {args.model_name}')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    add_prefix = True

    if args.specify_adafactor_lr == 'True':
        specify_adafactor_lr = True
    elif args.specify_adafactor_lr == 'False':
        specify_adafactor_lr = False
    else:
        raise Exception('Wrong Specify LR Mode!!!')

    log.info('Initialize dataclass')
    
    data = NLIdata( args.model_name, tokenizer, args.data_path_prefix,  args.ckpt_save_path)

    model = load_model(args, data, cuda_available)
    optimizer, scheduler = load_optimizer(model, args,  specify_adafactor_lr)
    max_accuracy = float("-inf")  
    for epoch in range(args.epoch_num):
        log.info(f'------------------------------Epoch {epoch}--------------------------------------')
        train_loss = train(args,model,optimizer, scheduler,specify_adafactor_lr, data,log, cuda_available, device)
        accuracy = evaluate(args, model,data,log,cuda_available, device)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            save_str = 'dev_joint_accuracy_{}'.format(round(accuracy*100,2))
            save_result(epoch, model, save_str, accuracy)
        log.info ('In the epoch {}, Currnt joint accuracy is {}, best joint accuracy is {}'.format(epoch, round(accuracy*100, 2), round(max_accuracy*100, 2)))
        


