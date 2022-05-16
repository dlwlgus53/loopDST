from queue import PriorityQueue
import os
import sys, pdb
import json
import torch
import random
import argparse
import operator
import torch.nn as nn
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
import time

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import logging
import logging.handlers
import copy

def train(args, model,optimizer, scheduler,specify_adafactor_lr, data,log, cuda_available, device):
    log.info('Training Session Start')
    model.train()
    train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train')
    train_batch_num_per_epoch = int(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))
    epoch_step, train_loss = 0, 0.
    for idx,train_batch in enumerate(train_iterator):
        if idx %300 ==0:
            log.info(f'{idx*100/train_batch_num_per_epoch:.4f} % works')
        
        one_train_input_batch, one_train_output_batch = train_batch
        if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
        train_batch_src_tensor, train_batch_src_mask, labels = \
        data.parse_batch_tensor(train_batch)
        if idx %300 ==0:
            log.info("text")
            log.info(f'{data.tokenizer.decode(train_batch_src_tensor[0])}')
            log.info("label")
            log.info(f'{data.tokenizer.decode(labels[0])}')
            
            
        if cuda_available:
            train_batch_src_tensor = train_batch_src_tensor.to(device)
            train_batch_src_mask = train_batch_src_mask.to(device)
            labels = labels.to(device)
        loss = model.module.classification(train_batch_src_tensor, train_batch_src_mask, labels)
        if idx %50 ==0:
            log.info(f"loss : {loss:.4f}")
    
        
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        epoch_step += 1
        if (epoch_step+1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1) == train_batch_num_per_epoch:
            optimizer.step()
            if args.optimizer_name == 'adafactor' and not specify_adafactor_lr:
                scheduler.step()
            elif args.optimizer_name == 'adam':
                scheduler.step() # only update learning rate when using adam
            else:
                pass
            optimizer.zero_grad()
    train_loss = train_loss / train_batch_num_per_epoch
    return train_loss


def evaluate(args, model,data,log, cuda_available, device):
    correct = 0
    all_number =0 
    log.info('Evalation Session Start')
    dev_batch_num_per_epoch = int(data.dev_num / (args.number_of_gpu * args.batch_size_per_gpu))
    
    model.eval()
    with torch.no_grad():
        dev_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='valid')

        for idx, batch in enumerate(dev_iterator):
            if idx %300 ==0:
                log.info(f'{idx*100/dev_batch_num_per_epoch:.4f} % works')
            input_batch, output_batch = batch
            src_tensor, src_mask, labels = \
            data.parse_batch_tensor(batch)
            
            if cuda_available:
                src_tensor = src_tensor.to(device)
                src_mask = src_mask.to(device)
            if len(input_batch) == 0 or len(output_batch) == 0: break
            output_results = model.module.classification_generate(src_tensor, src_mask)
            
            if idx %300 ==0:
                log.info("text")
                log.info(f'{data.tokenizer.decode(src_tensor[0])}')
                log.info("label")
                log.info(f'{data.tokenizer.decode(labels[0])}:')
                log.info("predict")
                log.info(output_results[0])
            
            
            for label_idx, output in enumerate(output_results):
                all_number +=1
                if output == data.tokenizer.decode(labels[label_idx]):
                    correct +=1
                           
    
    return correct/all_number

