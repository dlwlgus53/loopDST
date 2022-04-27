from queue import PriorityQueue
import os
import sys, pdb
import json
import torch
import random
import argparse
import operator
import progressbar
import torch.nn as nn
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
from inference_utlis import batch_generate

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import logging
import logging.handlers
import copy


def tagging(args,model,data,log, cuda_available, device):
    confidence_que = PriorityQueue()
    log.info('TAGGING Session Start')
    model.eval()
    with torch.no_grad():
        tagging_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='tagging')
        tagging_batch_num_per_epoch = int(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))
        if args.use_progress: p = progressbar.ProgressBar(tagging_batch_num_per_epoch)
        if args.use_progress: p.start()
        p_tagging_idx = 0
        
        for train_batch, dial_turn_key_batch in tagging_iterator:
            p_tagging_idx += 1
            if args.use_progress: p.update(p_tagging_idx)
            else:
                if p_tagging_idx%10 == 0: log.info(f'Tagged {p_tagging_idx* 100/tagging_batch_num_per_epoch:.2f} %')       
            one_train_input_batch, one_train_output_batch = train_batch
            if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break

            train_batch_src_tensor, train_batch_src_mask, _, _ = \
            data.parse_batch_tensor(train_batch)
            if cuda_available:
                src_input = train_batch_src_tensor.to(device)
                src_mask = train_batch_src_mask.to(device)
            tagging_batch_parse_dict, confidence_list = model.module.tagging(src_input, src_mask)   
            for predict_result,dial_turn_key, confidence in zip(tagging_batch_parse_dict, dial_turn_key_batch, confidence_list):
                confidence_que.put((-confidence, (dial_turn_key , '<sos_b> ' + predict_result + ' <eos_b>')))
        if args.use_progress: p.finish()
    cnt =0
    labeled_json_path = args.data_path_prefix + '/labeled.json'
    labeled_data = data.labeled_data
    
    qsize = confidence_que.qsize()
    while confidence_que.empty() != True:
        cnt +=1
        key, value = confidence_que.get()[1]
        assert key not in labeled_data
        labeled_data[key] = value
        if cnt>qsize*args.confidence_percent:
            break
    with open(labeled_json_path, 'w') as outfile:
        json.dump(labeled_data, outfile, indent=4)
    data.update_labeled_data()
    log.info(f"Saved tagged data until confidence {float(args.confidence_percent)*100} %")
    
def train(args, model,optimizer, scheduler,specify_adafactor_lr, data,log, cuda_available, device):
    
    log.info('Training Session Start')
    model.train()
    
    if args.loop == 0:
        train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train')
    else:
        train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train_loop')
        
    train_batch_num_per_epoch = int(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))
    if args.use_progress: p = progressbar.ProgressBar(train_batch_num_per_epoch)
    if args.use_progress: p.start()
    p_train_idx = 0
    epoch_step, train_loss = 0, 0.
    for train_batch, _ in train_iterator:
        p_train_idx += 1
        if p_train_idx == 1:
            train_batch_num_per_epoch = int(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))

        if args.use_progress: p.update(p_train_idx)
        else:
            if p_train_idx%100 == 0: log.info(f'Training {p_train_idx*100/train_batch_num_per_epoch:.2f} %')
        one_train_input_batch, one_train_output_batch = train_batch
        if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
        train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
        data.parse_batch_tensor(train_batch)
        if cuda_available:
            train_batch_src_tensor = train_batch_src_tensor.to(device)
            train_batch_src_mask = train_batch_src_mask.to(device)
            train_batch_input = train_batch_input.to(device)
            train_batch_labels = train_batch_labels.to(device)
        try:
            loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
        except:
            pdb.set_trace()
        loss = loss.mean()
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
    if args.use_progress: p.finish()
    train_loss = train_loss / train_batch_num_per_epoch
    
    return train_loss
def evaluate(args,model,data,log, cuda_available, device):
    log.info('Evalation Session Start')
    model.eval()
    with torch.no_grad():
        dev_batch_list = \
        data.build_all_evaluation_batch_list(eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='dev')
        dev_batch_num_per_epoch = len(dev_batch_list)
        if args.use_progress: p = progressbar.ProgressBar(dev_batch_num_per_epoch)
        log.info ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
        if args.use_progress: p.start()
        
        all_dev_result = []
        for p_dev_idx in range(dev_batch_num_per_epoch):
            if args.use_progress: p.update(p_dev_idx)
            else:
                if p_dev_idx%100 == 0: log.info(f'Evaluation {p_dev_idx*100/dev_batch_num_per_epoch:.2f} %')
            one_inference_batch = dev_batch_list[p_dev_idx]
            dev_batch_parse_dict = batch_generate(model, one_inference_batch, data)
            for item in dev_batch_parse_dict:
                all_dev_result.append(item)
        if args.use_progress: p.finish()

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
