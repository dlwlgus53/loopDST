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
from inference_utlis import batch_generate
import time
import copy
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import logging
import logging.handlers
import copy

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def tagging(args,model,data,log, cuda_available, device):
    confidence_que = PriorityQueue()
    log.info('TAGGING Session Start')
    model.eval()
    with torch.no_grad():

        tagging_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='tagging')
      
        for idx, (tagging_batch, dial_turn_key_batch) in enumerate(tagging_iterator):
            if idx  == 0:
                tagging_data_num = len(data.tagging_data_list)
                tagging_batch_num_per_epoch = int(tagging_data_num / (args.number_of_gpu * args.batch_size_per_gpu))+1
                log.info(f"unlabeled data num : {tagging_data_num}")

            if idx%args.log_interval == 0: log.info(f'Tagged {idx* 100/tagging_batch_num_per_epoch:.2f} %')       
            one_train_input_batch, one_train_output_batch = tagging_batch
            if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break

            tagging_batch_src_tensor, tagging_batch_src_mask, _, _ = \
            data.parse_batch_tensor(tagging_batch)
            if cuda_available:
                src_input = tagging_batch_src_tensor.to(device)
                src_mask = tagging_batch_src_mask.to(device)
            tagging_batch_parse_dict, confidence_list = model.module.tagging(src_input, src_mask)   
            for predict_result,dial_turn_key, confidence in zip(tagging_batch_parse_dict, dial_turn_key_batch, confidence_list):
                confidence_que.put((-confidence, (dial_turn_key , '<sos_b> ' + predict_result + ' <eos_b>')))
    
    labeled_cnt =0
    
    prev_labeled_data_len = len(data.labeled_data)
    labeled_data = data.labeled_data
    
    qsize = confidence_que.qsize()
    while confidence_que.empty() != True:
        labeled_cnt +=1
        key, value = confidence_que.get()[1]
        assert key not in labeled_data
        labeled_data[key] = value
        if labeled_cnt>qsize*args.confidence_percent:
            break
    # init_labeled_data = data.get_init_data()
    # labeled_data = merge_two_dicts(labeled_data, init_labeled_data)
    data.update_labeled_data(labeled_data)
    
    log.info(f"prior labeld data: {prev_labeled_data_len} unlabeld data: {tagging_data_num} saving :{labeled_cnt}")
    log.info(f"updated tagged data: {len(data.labeled_data)}")
    
def train(args, model,optimizer, scheduler, data,log, cuda_available, device):
    model.train()
    train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train_loop')
    epoch_step, train_loss = 0, 0.
    for idx, (train_batch, _) in enumerate(train_iterator):
        if idx == 0:
            train_num = len(data.train_data_list)
            train_batch_num_per_epoch = int(train_num / (args.number_of_gpu * args.batch_size_per_gpu))+1
        idx += 1

        if idx%100 == 0: log.info(f'Training {idx*100/train_batch_num_per_epoch:.2f} %')
        one_train_input_batch, one_train_output_batch = train_batch
        if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
        train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
        data.parse_batch_tensor(train_batch)
        if cuda_available:
            train_batch_src_tensor = train_batch_src_tensor.to(device)
            train_batch_src_mask = train_batch_src_mask.to(device)
            train_batch_input = train_batch_input.to(device)
            train_batch_labels = train_batch_labels.to(device)
        loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        epoch_step += 1

        if (epoch_step+1) % args.gradient_accumulation_steps == 0 or (epoch_step + 1) == train_batch_num_per_epoch:
            optimizer.step()
    train_loss = train_loss / train_batch_num_per_epoch
    
    return train_loss

def evaluate(args,model,data,log, cuda_available, device):
    log.info('Evalation Session Start')
    model.eval()
    with torch.no_grad():
        dev_batch_list = \
        data.build_all_evaluation_batch_list(eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='dev')
        dev_batch_num_per_epoch = len(dev_batch_list)
        log.info ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
        
        all_dev_result = []
        for p_dev_idx in range(dev_batch_num_per_epoch):
            if p_dev_idx%args.log_interval == 0: log.info(f'Evaluation {p_dev_idx*100/dev_batch_num_per_epoch:.2f} %')
            one_inference_batch = dev_batch_list[p_dev_idx]
            dev_batch_parse_dict = batch_generate(model, one_inference_batch, data)
            for item in dev_batch_parse_dict:
                all_dev_result.append(item)
        from compute_joint_acc import compute_jacc
        all_dev_result = zip_result(all_dev_result)
        dev_score = compute_jacc(data=all_dev_result) * 100
    
    return all_dev_result, dev_score


def zip_result(prediction):
    result = {}
    for turn in prediction:
        dial_id = turn['dial_id']
        turn_idx = turn['turn_num']
        try:
            result[dial_id][turn_idx] = turn
        except KeyError:
            result[dial_id] = {}
            result[dial_id][turn_idx] = turn
    return result
