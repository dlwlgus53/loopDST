# from trainer import  train, evaluate
import torch
import torch.nn as nn
import pdb
        
def train(args,model,optimizer,data,log, cuda_available, device):
    model.train()
    train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode = "train")
    train_loss = 0.
    
    for idx, train_batch in enumerate(train_iterator):
        if idx == 0:
            train_num = len(data.train_data_list)
            train_batch_num_per_epoch = int(train_num / (args.number_of_gpu * args.batch_size_per_gpu))+1
        idx += 1
        one_train_input_batch, one_train_output_batch = train_batch
        if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
        source_input, _, target_input, _ = \
        data.parse_batch_tensor(train_batch)
        if cuda_available:
            input_ids = source_input.to(device)
            labels = target_input.to(device)
        outputs = model(input_ids = input_ids, labels = labels)
        loss = outputs.loss.mean()
        loss.backward()
        train_loss += loss.item()
        if idx%50 == 0: 
            logit_result = torch.max(outputs.logits[0].detach().cpu(),1).indices
            log.info(f'{idx*100/train_batch_num_per_epoch:.2f}% done. loss > {train_loss/idx:.4f}')
            log.info(f'INPUT : {train_batch[0][0]}')
            log.info(f'LABEL : {train_batch[1][0]}')
            log.info(f'OUTPUT : {data.tokenizer.decode(logit_result, skip_special_tokens = True)}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (idx+1) % args.gradient_accumulation_steps == 0 or (idx + 1) == train_batch_num_per_epoch:
            optimizer.step()
    train_loss = train_loss / train_batch_num_per_epoch
    return train_loss

def evaluate(args,model,data,log, cuda_available, device):
    model.eval()
    dev_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode = "dev")
    dev_loss =0. 
    
    with torch.no_grad():
        for idx, dev_batch in enumerate(dev_iterator):
            if idx == 0:
                dev_num = len(data.dev_data_list)
                dev_batch_num_per_epoch = int(dev_num / (args.number_of_gpu * args.batch_size_per_gpu))+1
            idx += 1
            if idx%50 == 0: log.info(f'{idx*100/dev_batch_num_per_epoch:.2f} %')
            one_dev_input_batch, one_dev_output_batch = dev_batch
            if len(one_dev_input_batch) == 0 or len(one_dev_output_batch) == 0: break
            source_input, _, target_input, _ = \
            data.parse_batch_tensor(dev_batch)
            if cuda_available:
                input_ids = source_input.to(device)
                labels = target_input.to(device)
            outputs = model(input_ids = input_ids, labels = labels)
            dev_loss += outputs.loss.mean().item()
            
    return dev_loss/idx

