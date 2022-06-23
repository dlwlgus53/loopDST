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
import logging
import logging.handlers
from trainer_part import  evaluate
from modelling.T5Model import T5Gen_Model
from dataclass_part import DSTMultiWozData

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
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--model_name', type=str, help='t5-base or t5-large or facebook/bart-base or facebook/bart-large')
    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    parser.add_argument('--init_label_path', type=str, default='None', help='the path that stores pretrained checkpoint.')
    parser.add_argument("--aug_model_path", type=str, default="None", help="confidence percent")
    
    # training configuration
    parser.add_argument('--optimizer_name', default='adafactor', type=str, help='which optimizer to use during training, adam or adafactor')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    parser.add_argument("--epoch_num", default=60, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--mini_epoch",  default=5, type=int,help="mini epoch")
    parser.add_argument("--aug_epoch", default = 1,  type=int, help="use augment or not")
    parser.add_argument("--aug_num", default = 2,  type=int, help="use augment or not")
    
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--eval_batch_size_per_gpu", type=int, default=8, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--confidence_percent", type=float, default=0.5, help="confidence percent")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="confidence percent")
    
    
    parser.add_argument("--debugging", type=int, default=0, help="debugging going small")
    parser.add_argument("--log_interval", type=int, default=1000, help="mini epoch")
    parser.add_argument("--aug_method", type=int, help="use augment or not")
    
    return parser.parse_args()

def load_model(args, data, cuda_available, load_pretrained = True):
    if args.pretrained_path != 'None' and load_pretrained:
        model = T5Gen_Model(args.pretrained_path, data.tokenizer, data.special_token_list, dropout=args.dropout, 
            add_special_decoder_token=add_special_decoder_token, is_training=True)
    else:
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
    return model


if __name__ == '__main__':
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
    fileHandler = logging.FileHandler(f'{args.ckpt_save_path}log.txt')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    
    log.info('seed setting')
    log.info(args)
    
    init_experiment(args)
    
    
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    if args.pretrained_path != 'None':
        log.info (f'Loading Pretrained Tokenizer... {args.pretrained_path}')
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    else:
        log.info ('Loading from internet {args.model_name}')
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    add_special_decoder_token = True

    log.info('Initialize dataclass')
    
    data = DSTMultiWozData(args.model_name, tokenizer, args.data_path_prefix,  args.ckpt_save_path, init_label_path = args.init_label_path, \
        log_path = f'{args.ckpt_save_path}log.txt', shuffle_mode=args.shuffle_mode, \
          debugging = args.debugging)
    
    if args.aug_method in [1,2,3,4,6]:
        pre_trainer = Aug_training(args.aug_method, args.aug_num, args.aug_rate,\
            data, 'cuda', log, args.log_interval, args.eval_batch_size_per_gpu)
    else:
        pre_trainer = Aug_training(args.aug_method, args.aug_num, args.aug_rate,\
            data, 'cuda', log, args.log_interval, args.eval_batch_size_per_gpu, model_path = args.aug_model_path)
    


    log.info("load teacher model")
    model = load_model(args, data, cuda_available)
    optimizer, scheduler = load_optimizer(model, args)
    min_dev_loss = 1e10
    max_dev_score, max_dev_str = 0., ''
    score_list = ["Best scores"]
    
    for epoch in range(args.epoch_num):
        log.info(f'------------------------------Epoch {epoch}--------------------------------------')
        log_sentence.append(f"Epoch {epoch}")
        log.info(f"Epoch {epoch} Tagging start")
        model= load_model(args, data, cuda_available, load_pretrained = False)

        if args.aug_method:
            log.info("aug data training")
            aug_train, aug_dev = pre_trainer.augment()
            student = pre_trainer.train(args, aug_train, aug_dev,student, args.aug_epoch, optimizer, scheduler)
            
        mini_best_result, mini_best_str, mini_score_list = 0, '', ['mini epoch']
        for mini_epoch in range(args.mini_epoch):
            log.info(f"Epoch {epoch}-{mini_epoch} training start")
            train_loss = train(args,student,optimizer, scheduler, data,log, cuda_available, device, mode = 'train_loop')
            log.info (f'Epoch {epoch}-{mini_epoch} total training loss is %5f' % (train_loss))
            
            log.info (f'Epoch {epoch}-{mini_epoch} evaluate start')
            all_dev_result, dev_score = evaluate(args,student,data,log, cuda_available, device, mode = 'dev_loop')
            log.info (f'Epoch {epoch}-{mini_epoch} JGA is {dev_score}')
            mini_score_list.append(f'{dev_score:.2f}')
            
            if dev_score > mini_best_result:
                model = student
                one_dev_str = 'dev_joint_accuracy_{}'.format(round(dev_score,2))
                mini_best_str = one_dev_str
                mini_best_result = dev_score
                save_result(epoch, model, mini_best_str, mini_best_result)
            log.info ('In the mini epoch {}, Currnt joint accuracy is {}, best joint accuracy is {}'.format(mini_epoch, round(dev_score, 2), round(mini_best_result, 2)))
        
        log_sentence.append(" ".join(mini_score_list))
        score_list.append(f'{mini_best_result:.2f}')
        

    
    log_sentence.append(" ".join(score_list))    
    log.info(score_list)
    
    with open(f'{args.ckpt_save_path}log.txt', 'a') as f:
        for item in log_sentence:
            f.write("%s\n" % item)
            
        
    
    
