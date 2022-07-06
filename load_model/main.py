import os
import torch
import argparse
import torch.nn as nn
from operator import itemgetter
from modelling.T5Model import T5Gen_Model
import random
from dataclass import DSTMultiWozData


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
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-base or t5-large or facebook/bart-base or facebook/bart-large')
    parser.add_argument('--pretrained_path', type=str, default='None', help='the path that stores pretrained checkpoint.')

    parser.add_argument('--optimizer_name', default='adafactor', type=str, help='which optimizer to use during training, adam or adafactor')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    return parser.parse_args()

def get_optimizers(model, args):
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
    if args.optimizer_name == 'adam':
        pass
    elif args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule
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

    return optimizer, scheduler

def load_model(args, data, cuda_available, load_pretrained = True):
    device = torch.device('cuda')
    add_special_decoder_token = True
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


def load_optimizer(model, args):
    optimizer, scheduler = get_optimizers(model, args)
    optimizer.zero_grad()
    return optimizer, scheduler
    

def save_result(epoch, model, one_dev_str,all_dev_result):
    model_save_path = args.ckpt_save_path + '/epoch_' + str(epoch) + '_' + one_dev_str

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
            os.system('rm -r ' + one_folder_name)

def get_best_model_path(ckpt_path, num):
    model_names = []
    for fname in os.listdir(ckpt_path):
        if fname.startswith(f'epoch_'):
            model_names.append(fname)
    
    model_names = sorted(model_names, key = lambda x : float(x.split("_")[-1]), reverse=True)
    return model_names[:num]
    
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise

if __name__ == '__main__':
    log_sentence = []
    # MAKE FOLDER
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
 
    args = parse_config()
    
    
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer
    if args.pretrained_path != 'None':
        print (f'Loading Pretrained Tokenizer... {args.pretrained_path}')
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    else:
        print (f'Loading from huggingface {args.model_name}')
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)


    data = DSTMultiWozData(args.model_name, tokenizer, args.data_path_prefix,  shuffle_mode='shuffle_session_level', init_label_path = None, save_label_path = None,
        data_mode='train', add_prefix=True, add_special_decoder_token=True, train_data_ratio=1.0)
    model = load_model(args, data, cuda_available)
    