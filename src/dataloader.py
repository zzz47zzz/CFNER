
import torch
import torch.nn as nn
import os
import numpy as np
import random
import logging
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from copy import deepcopy

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_params

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index
max_seq_length = params.max_seq_length

# domain name to entity list
domain2entity={
    # #conll2003=4
    'conll2003': ['location', 'misc', 'organisation', 'person'], 
    # #ontonotes5=18
    'ontonotes5': ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'],
    # #i2b2=16 
    'i2b2': ['AGE', 'CITY', 'COUNTRY', 'DATE', 'DOCTOR', 'HOSPITAL', 'IDNUM', 'MEDICALRECORD', 'ORGANIZATION', 'PATIENT', 'PHONE', 'PROFESSION', 'STATE', 'STREET', 'USERNAME', 'ZIP'],
    # #ontonotes4=18
    'ontonotes4': ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'],   
    # # #i2b2=12
    # 'i2b2': ['AGE', 'CITY', 'COUNTRY', 'DATE', 'DOCTOR', 'HOSPITAL', 'IDNUM', 'MEDICALRECORD', 'PATIENT', 'PROFESSION', 'STATE', 'ZIP'],
    # 'i2b2': ['AGE', 'BIOID', 'CITY', 'COUNTRY', 'DATE', 'DEVICE', 'DOCTOR', 'EMAIL', 'FAX', 'HEALTHPLAN', 'HOSPITAL', 'IDNUM', 'LOCATION_OTHER', 'MEDICALRECORD', 'ORGANIZATION', 'PATIENT', 'PHONE', 'PROFESSION', 'STATE', 'STREET', 'URL', 'USERNAME', 'ZIP'],
    # 'wnut17': ['corporation', 'creative-work', 'group', 'location', 'person', 'product'],
}

def print_split_data_statistic(datapath, phase, entity_list, nb_class_fg, nb_class_pg, schema):
    label_list = get_default_label_list(entity_list, schema=schema)
    if isinstance(datapath,list):
        data_ckpt = os.path.join(datapath[0],'train_fg_%d_pg_%d.pth'%(nb_class_fg,nb_class_pg))
        # data_ckpt = os.path.join(datapath[0],'train_fg_%d_pg_%d_random.pth'%(nb_class_fg,nb_class_pg))
    else:
        data_ckpt = os.path.join(datapath,'train_fg_%d_pg_%d.pth'%(nb_class_fg,nb_class_pg))
        # data_ckpt = os.path.join(datapath,'train_fg_%d_pg_%d_random.pth'%(nb_class_fg,nb_class_pg))
    print('Loading data from %s...'%data_ckpt)
    with open(data_ckpt, "rb") as f:
        inputs_dict, y_dict = pickle.load(f)
    for k,v in y_dict.items():
        print('Data split %s:'%k)
        print(sorted(get_label_distribution(v,label_list,count=True).items(),key=lambda x:x[0]))

def spilt_dataset_random(datapath, phase, entity_list, nb_class_fg, nb_class_pg, schema):
    label_list = get_default_label_list(entity_list, schema=schema)
    all_x, all_y = read_ner(datapath, phase, label_list)
    entity_idx = 0
    split_name_list = []
    entity_2_split = {}
    split_cnt_dict = {}
    inputs_dict, y_dict = {}, {}

    num_all_samples = len(all_x)
    while entity_idx<len(entity_list):
        if entity_idx==0:
            split_name = '%d_%d'%(entity_idx, entity_idx+nb_class_fg-1)
            split_name_list.append(split_name)
            split_cnt_dict[split_name] = int(num_all_samples*(nb_class_fg/len(entity_list)))
            for i in range(entity_idx,entity_idx+nb_class_fg):
                entity_2_split[i]=split_name
            entity_idx = entity_idx+nb_class_fg
        else:
            split_name = '%d_%d'%(entity_idx, entity_idx+nb_class_pg-1)
            split_name_list.append(split_name)
            split_cnt_dict[split_name] = int(num_all_samples*(nb_class_pg/len(entity_list)))
            for i in range(entity_idx,entity_idx+nb_class_pg):
                entity_2_split[i]=split_name
            entity_idx = entity_idx+nb_class_pg
    
    for split_name in split_name_list:
        inputs_dict[split_name], y_dict[split_name] = [], []

    for x_sent, y_sent in zip(all_x, all_y):
        not_full_split = []
        for i in split_name_list:
            if split_cnt_dict[i]>0:
                not_full_split.append(i)
        if len(not_full_split)==0:
            break
        split_name = np.random.choice(not_full_split)
        inputs_dict[split_name].append(x_sent)
        y_dict[split_name].append(y_sent)
        split_cnt_dict[split_name] -= 1

    if isinstance(datapath,list):
        data_ckpt = os.path.join(datapath[0],'train_fg_%d_pg_%d_random.pth'%(nb_class_fg,nb_class_pg))
    else:
        data_ckpt = os.path.join(datapath,'train_fg_%d_pg_%d_random.pth'%(nb_class_fg,nb_class_pg))
    logger.info('Saving data to %s...'%data_ckpt)
    with open(data_ckpt, "wb") as f:
        pickle.dump((inputs_dict, y_dict), f)


def spilt_dataset(datapath, phase, entity_list, nb_class_fg, nb_class_pg, schema):
    label_list = get_default_label_list(entity_list, schema=schema)
    all_x, all_y = read_ner(datapath, phase, label_list)
    entity_idx = 0
    split_name_list = []
    entity_2_split = {}
    split_cnt_dict = {}
    inputs_dict, y_dict = {}, {}

    num_all_samples = len(all_x)
    while entity_idx<len(entity_list):
        if entity_idx==0:
            split_name = '%d_%d'%(entity_idx, entity_idx+nb_class_fg-1)
            split_name_list.append(split_name)
            split_cnt_dict[split_name] = int(num_all_samples*(nb_class_fg/len(entity_list)))
            for i in range(entity_idx,entity_idx+nb_class_fg):
                entity_2_split[i]=split_name
            entity_idx = entity_idx+nb_class_fg
        else:
            split_name = '%d_%d'%(entity_idx, entity_idx+nb_class_pg-1)
            split_name_list.append(split_name)
            split_cnt_dict[split_name] = int(num_all_samples*(nb_class_pg/len(entity_list)))
            for i in range(entity_idx,entity_idx+nb_class_pg):
                entity_2_split[i]=split_name
            entity_idx = entity_idx+nb_class_pg
    
    for split_name in split_name_list:
        inputs_dict[split_name], y_dict[split_name] = [], []

    # Sorted by the frequency
    label_count_train = get_label_distribution(all_y, label_list, count=True)
    # entity_label_list = sorted(label_count_train, key=lambda x:label_count_train[x])

    for x_sent, y_sent in zip(all_x, all_y):
        entity_set = list(set([(i-1)//(len(schema)-1) for i in list(set(y_sent)-set([0,-100]))]))
        sorted_entity_set = sorted(entity_set, key=lambda x:label_count_train[entity_list[x]])
        is_break = False
        for entity_idx in sorted_entity_set:
            split_name = entity_2_split[entity_idx]
            if split_cnt_dict[split_name]>0:
                inputs_dict[split_name].append(x_sent)
                y_dict[split_name].append(y_sent)
                split_cnt_dict[split_name] -= 1
                is_break = True
                break
        if not is_break:
            not_full_split = []
            for k,v in split_cnt_dict.items():
                if v>0:
                    not_full_split.append(k)
            if len(not_full_split)>0:
                split_name = np.random.choice(not_full_split)
                inputs_dict[split_name].append(x_sent)
                y_dict[split_name].append(y_sent)
                split_cnt_dict[split_name] -= 1
                
    # print([len(v) for k,v in inputs_dict.items()])
    # print([len(v) for k,v in y_dict.items()])

    if isinstance(datapath,list):
        data_ckpt = os.path.join(datapath[0],'train_fg_%d_pg_%d.pth'%(nb_class_fg,nb_class_pg))
    else:
        data_ckpt = os.path.join(datapath,'train_fg_%d_pg_%d.pth'%(nb_class_fg,nb_class_pg))
    logger.info('Saving data to %s...'%data_ckpt)
    with open(data_ckpt, "wb") as f:
        pickle.dump((inputs_dict, y_dict), f)

def convert_BIOES_to_BIO(in_datapath, out_datapath):
    with open(out_datapath, "w", encoding='utf-8') as f_out:
        with open(in_datapath, "r", encoding="utf-8") as f_in:
            for i, line in enumerate(f_in):
                line = line.strip()
                splits = line.split()
                if line=='':
                    f_out.write('\n')
                    continue
                else:
                    label = splits[1]
                    if 'E-' in line:
                        line = line.replace('E-','I-')
                    elif 'S-' in line:
                        line = line.replace('S-','B-')
                    f_out.write(line+'\n')

def get_entity_list(datapth):
    label_list=[]
    with open(datapth, "r", encoding="utf-8") as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            splits = line.split()
            if line=="":
                continue
            label_list.append(splits[1])
    # Contains only entity type w/o B-/I-
    label_list = sorted(list(set(label_list)))
    entity_list = []
    for l in label_list:
        if ('B-' in l) or ('I-' in l) or ('E-' in l) or ('S-' in l): 
            entity_list.append(l[2:])
    entity_list = list(set(entity_list))
    print("entity_list = %s"%str(entity_list))
    return entity_list

def get_default_label_list(entity_list, schema='BIO'):
    default_label_list = []
    default_label_list.append('O')
    if schema=='IO':
        for e in entity_list:
            default_label_list.append('I-'+str(e))
    elif schema=='BIO':
        for e in entity_list:
            default_label_list.append('B-'+str(e))
            default_label_list.append('I-'+str(e))
    elif schema=='BIOES':
        for e in entity_list:
            default_label_list.append('B-'+str(e))
            default_label_list.append('I-'+str(e))
            default_label_list.append('E-'+str(e))
            default_label_list.append('S-'+str(e))
    return default_label_list

def read_ner(datapath, phase, label_list):
    if isinstance(datapath,list):
        if len(datapath)>1 and phase!="train":
            logger.warning("In %s phase, more than one domain data are combined!!!"%(phase))
        data_path_lst = [os.path.join(_path, phase+".txt") for _path in datapath]
    else:
        data_path_lst = [os.path.join(datapath, phase+".txt")]
    
    inputs, ys = [], []
    for _datapath in data_path_lst:
        _inputs, _ys = [], []
        data_ckpt = _datapath[:-4]+'.pth'
        if os.path.isfile(data_ckpt):
            logger.info('Loading data from %s...'%data_ckpt)
            with open(data_ckpt, "rb") as f:
                _inputs, _ys = pickle.load(f)
            inputs.append(_inputs)
            ys.append(_ys)
            continue
        with open(_datapath, "r", encoding="utf-8") as fr:
            token_list, y_list = [], []
            for i, line in enumerate(fr):
                line = line.strip() 
                if line == "":
                    if len(token_list) > 0:
                        assert len(token_list) == len(y_list)
                        _inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                        _ys.append([pad_token_label_id] + y_list + [pad_token_label_id])

                    token_list, y_list = [], []
                    continue
                splits = line.split()
                token = splits[0]
                label = splits[1]

                subs_ = auto_tokenizer.tokenize(token)
                if len(subs_) > 0:
                    y_list.extend([label_list.index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                    token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
                else:
                    print("length of subwords for %s is zero; its label is %s" % (token, label))
        inputs.append(_inputs)
        ys.append(_ys)
        with open(data_ckpt, "wb") as f:
            pickle.dump((_inputs, _ys), f)

    # combine data from different domains (only for training data)
    sample_cnt_lst = [len(_ys) for _ys in ys]
    max_cnt = max(sample_cnt_lst)
    inputs_all, ys_all = [], []
    for _inputs, _ys in zip(inputs, ys):
        ratio = int(max_cnt/len(_ys))
        inputs_all.extend(_inputs*ratio) 
        ys_all.extend(_ys*ratio)

    return inputs_all, ys_all

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, ys):
        self.X = inputs
        self.y = ys
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    for i, (seq, y_) in enumerate(zip(X, y)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)

    if max_lengths>max_seq_length:
        padded_seqs, padded_y = padded_seqs[:,:max_seq_length], padded_y[:,:max_seq_length],

    return padded_seqs, padded_y

def entity_sampler(inputs_train, y_train, label_list, unbalanced=False, n_samples=10):

    if n_samples==-1 and not unbalanced:
        return inputs_train, y_train

    n_samples_list = []
    # Sorted by the frequency
    label_count_train = get_label_distribution(y_train, label_list, count=True)
    entity_label_list = sorted(label_count_train, key=lambda x:label_count_train[x])
    max_entity_count = label_count_train[entity_label_list[-1]]
    min_entity_count = label_count_train[entity_label_list[0]]
    imbalance_ratio = min_entity_count/max_entity_count
    entity_count_list = [0]*len(entity_label_list)
    n_labels = len(entity_label_list)
    select_index = []

    # Generate sample number list
    if unbalanced:
        # Satisfy the tail classes first
        n_samples_list = [int(max_entity_count*(imbalance_ratio**((i-1)/(n_labels-1))))\
                        for i in range(n_labels,0,-1)]
        # Satisfy the head classes first
        # entity_label_list = list(reversed(entity_label_list))
        # n_samples_list = [int(max_entity_count*(imbalance_ratio**(i/(n_labels-1))))\
        #                 for i in range(n_labels)]
    else:
        n_samples_list = [n_samples]*len(label_list)
    logger.info("Sample number list: %s" % n_samples_list)

    # Each entity 
    for i, entity in enumerate(entity_label_list):
        if len(select_index) >= len(inputs_train):
            logger.info("Break because not enough training samples!!!")
            break
        is_brk = False
        sample_index = 0
        # Loop until entity count is satisfied 
        while entity_count_list[i]<n_samples_list[i]: 
            sample_index += 1
            if sample_index>=len(inputs_train):
                is_brk = True
            # Sample until not in selected list
            while sample_index in select_index:
                sample_index += 1
                if sample_index>=len(inputs_train):
                    is_brk = True
            # Can not find more entity
            if is_brk:
                logger.info("Break because not enough %s entity samples!!!"%(entity))
                break
            # Select the sentence
            label_seq = y_train[sample_index]
            # Check if the entity in the sentence
            if label_list.index('B-'+entity) in label_seq:
                select_index.append(sample_index)
                # Update all counts and select this samples
                for j, _entity in enumerate(entity_label_list):
                    for _label in label_seq:
                        if label_list.index('B-'+_entity) == _label:
                            entity_count_list[j] += 1
    # Print the number for each entity
    label_count_dict = dict()
    for i, entity in enumerate(entity_label_list): 
        label_count_dict[entity] = entity_count_list[i]
    logger.info(label_count_dict)

    # Select the training samples by indexes
    inputs_train, y_train = np.array(inputs_train), np.array(y_train)
    inputs_train, y_train = inputs_train[select_index], y_train[select_index]
    inputs_train, y_train = list(inputs_train), list(y_train)

    return inputs_train, y_train

def get_label_distribution(y_lists,label_list, count=False):
    label_distribution = dict()
    count_tok_test = 0
    for y_list in y_lists:
        for y in y_list:
            if y != pad_token_label_id:
                label_name = label_list[y]
                if "B-" in label_name or "S-" in label_name:
                    count_tok_test += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distribution:
                        label_distribution[label_name] = 1
                    else:
                        label_distribution[label_name] += 1
    if count:
        return label_distribution
    else:
        for key in label_distribution:
            freq = label_distribution[key] / count_tok_test
            label_distribution[key] = round(freq, 2)
        return label_distribution

class NER_dataloader():
    def __init__(self, data_path, domain_name, batch_size, entity_list=[], n_samples=-1, is_filter_O=False, schema='BIO', is_load_disjoin_train=False):

        self.batch_size = batch_size
        self.is_filter_O = is_filter_O
        self.auto_tokenizer = auto_tokenizer
        self.schema = schema
        self.is_load_disjoin_train = is_load_disjoin_train

        # Get entity list
        if entity_list=="" or entity_list==[]:
            logger.info('Loading the default entity list from domain %s...'%domain_name)
            self.entity_list = domain2entity[domain_name]
        else:
            logger.info('Loading the pre-defined entity list...')
            entity_list = eval(entity_list)
            assert isinstance(entity_list,list), "Wrong entity_list type!!!"
            self.entity_list = entity_list

        # Get label list
        self.label_list = get_default_label_list(self.entity_list, schema=self.schema)
        self.O_index = self.label_list.index('O')
        logger.info('label_list = %s'%str(self.label_list))

        # Load data
        logger.info("Load training set data")
        if self.is_load_disjoin_train:
            if isinstance(data_path,list):
                train_data_ckpt = os.path.join(data_path[0],'train_fg_%d_pg_%d.pth'%(params.nb_class_fg,params.nb_class_pg))
                # train_data_ckpt = os.path.join(data_path[0],'train_fg_%d_pg_%d_random.pth'%(params.nb_class_fg,params.nb_class_pg))
            else:
                train_data_ckpt = os.path.join(data_path,'train_fg_%d_pg_%d.pth'%(params.nb_class_fg,params.nb_class_pg))
                # train_data_ckpt = os.path.join(data_path,'train_fg_%d_pg_%d_random.pth'%(params.nb_class_fg,params.nb_class_pg))
            logger.info('Loading data from %s...'%train_data_ckpt)
            with open(train_data_ckpt, "rb") as f:
                inputs_train_dict, y_train_dict = pickle.load(f)
            self.inputs_train_dict, self.y_train_dict = inputs_train_dict, y_train_dict
            logger.info("train size for each split: %s" % 
                        ([k+':'+str(len(v)) for k,v in inputs_train_dict.items()]))
        else:
            inputs_train, y_train = read_ner(data_path, 
                                            phase="train", 
                                            label_list=self.label_list)
            logger.info("label count for train set")
            _label_count_train = get_label_distribution(y_train,self.label_list,count=True)
            _label_count_train = sorted(_label_count_train.items(),key=lambda x:x[0])
            logger.info(_label_count_train)
            inputs_train, y_train = entity_sampler(inputs_train, y_train, self.label_list, unbalanced=False, n_samples=n_samples)
            self.inputs_train, self.y_train = inputs_train, y_train
            logger.info("train size: %d" % (len(inputs_train)))
            # logger.info("label distribution for train set")
            # logger.info(sorted(get_label_distribution(y_train,self.label_list,count=True).items(),key=lambda x:x[0]))

        # Only evaluate on the target domain (default in the first item)
        if isinstance(data_path, list):
            target_data_path = data_path[0]
        elif isinstance(data_path, str):
            target_data_path = data_path
        else:
            raise Exception('Data_path should be either a list or string!!!')
        logger.info("Load development set data")
        inputs_dev, y_dev = read_ner(target_data_path, 
                                    phase="dev", 
                                    label_list=self.label_list)
        logger.info("Load test set data")
        inputs_test, y_test = read_ner(target_data_path,
                                    phase="test",
                                    label_list=self.label_list)
        # Data statistic
        logger.info("label distribution for dev set")
        logger.info(sorted(get_label_distribution(y_dev,self.label_list,count=True).items(),key=lambda x:x[0]))
        logger.info("label distribution for test set")
        logger.info(sorted(get_label_distribution(y_test,self.label_list,count=True).items(),key=lambda x:x[0]))
        logger.info("dev size %d; test size: %d;" % (len(inputs_dev), len(inputs_test)))

        self.inputs_dev, self.y_dev = inputs_dev, y_dev
        self.inputs_test, self.y_test = inputs_test, y_test        

    def set_unseen_labels_to_O(self, y_train, seen_label_list):
        for i, y_lst in enumerate(y_train):
            for j, y in enumerate(y_lst):
                if y in [pad_token_label_id, self.label_list.index('O')]:
                    continue
                if not y in seen_label_list:
                    y_train[i][j] = self.label_list.index('O')
        return y_train

    def get_dataloader(self, first_N_classes=-1, select_entity_list=[], phase=['train','dev','test'], is_filter_O=None, filter_entity_list=[], is_ground_truth_train=False, reserved_ratio=0.0):

        if is_filter_O==None:
            is_filter_O = self.is_filter_O
        return_result = []

        if 'train' in phase:
            if self.is_load_disjoin_train:
                inputs_train_dict, y_train_dict = deepcopy(self.inputs_train_dict), deepcopy(self.y_train_dict)
            else:
                inputs_train, y_train = deepcopy(self.inputs_train), deepcopy(self.y_train)
        if 'dev' in phase:
            inputs_dev, y_dev = deepcopy(self.inputs_dev), deepcopy(self.y_dev)
        if 'test' in phase:
            inputs_test, y_test = deepcopy(self.inputs_test), deepcopy(self.y_test)

        if first_N_classes!=-1 or select_entity_list!=[]:
            # Get choosen label index corresponding to the original label list
            select_label_list = []
            select_label_list.append('O')
            if first_N_classes!=-1:
                assert  first_N_classes>0 and first_N_classes<=len(self.entity_list), "Invalid value of first_N_classes!!!"
                # logger.info("Select first %d classes: %s"%(first_N_classes,str(self.entity_list[:first_N_classes])))    
                for e in self.entity_list[:first_N_classes]:
                    if self.schema == 'IO':
                        select_label_list.append('I-'+str(e))  
                    elif self.schema == 'BIO':
                        select_label_list.append('B-'+str(e))
                        select_label_list.append('I-'+str(e)) 
                    elif self.schema == 'BIOES':
                        select_label_list.append('B-'+str(e))
                        select_label_list.append('I-'+str(e)) 
                        select_label_list.append('E-'+str(e)) 
                        select_label_list.append('S-'+str(e)) 
            elif select_entity_list!=[]:
                # logger.info("Select classes in %s !"%str(select_entity_list))
                for e in select_entity_list:
                    if self.schema == 'IO':
                        select_label_list.append('I-'+str(e))  
                    elif self.schema == 'BIO':
                        select_label_list.append('B-'+str(e))
                        select_label_list.append('I-'+str(e)) 
                    elif self.schema == 'BIOES':
                        select_label_list.append('B-'+str(e))
                        select_label_list.append('I-'+str(e)) 
                        select_label_list.append('E-'+str(e)) 
                        select_label_list.append('S-'+str(e)) 
            select_label_index = [self.label_list.index(l) for l in select_label_list]
            # Map the label to the seen class index
            if self.is_load_disjoin_train and 'train' in phase:
                select_label_index_wo_O = list(set(select_label_index)-set([0]))
                assert set(select_label_index_wo_O)==set(list(range(np.min(select_label_index_wo_O), np.max(select_label_index_wo_O)+1)))
                entity_index_min = (np.min(select_label_index_wo_O)-1)//(len(self.schema)-1)
                entity_index_max = np.max(select_label_index_wo_O)//(len(self.schema)-1)-1

                # concatnate all splits containing all the select entities
                inputs_train, y_train = [], []
                split_name_list = list(inputs_train_dict.keys())
                assert list(inputs_train_dict.keys())==list(y_train_dict.keys())

                # add reserved samples
                if reserved_ratio>0:
                    entity_idx_bg, entity_idx_ed = 0, -1
                    while entity_idx_ed!=entity_index_min-1:
                        is_find = False
                        for split_name in split_name_list:
                            if str(entity_idx_bg)+'_' in split_name:
                                entity_idx_ed = eval(split_name.split('_')[1])
                                is_find = True
                                break
                        assert is_find
                        tmp_split_name = '%d_%d'%(entity_idx_bg, entity_idx_ed)
                        reserved_number = int(len(inputs_train_dict[tmp_split_name])*reserved_ratio)
                        inputs_train.extend(inputs_train_dict[tmp_split_name][:reserved_number])
                        tmp_y_train = y_train_dict[tmp_split_name][:reserved_number]
                        # map to seen labels in the current step
                        tmp_select_label_list = list(range(1, 1+(entity_idx_ed+1)*(len(params.schema)-1)))
                        tmp_y_train = self.set_unseen_labels_to_O(tmp_y_train, tmp_select_label_list)

                        y_train.extend(tmp_y_train)
                        entity_idx_bg = entity_idx_ed+1

                # no reserved sample
                entity_idx_bg, entity_idx_ed = entity_index_min, -1
                while entity_idx_ed!=entity_index_max:
                    is_find = False
                    for split_name in split_name_list:
                        if str(entity_idx_bg)+'_' in split_name:
                            entity_idx_ed = eval(split_name.split('_')[1])
                            is_find = True
                            break
                    assert is_find
                    tmp_split_name = '%d_%d'%(entity_idx_bg, entity_idx_ed)
                    inputs_train.extend(inputs_train_dict[tmp_split_name])
                    tmp_y_train = y_train_dict[tmp_split_name]

                    # change the annotation
                    if params.extra_annotate_type=='none':
                        tmp_select_label_list = list(range(1+entity_idx_bg*(len(params.schema)-1), 1+(entity_idx_ed+1)*(len(params.schema)-1)))
                        tmp_y_train = self.set_unseen_labels_to_O(tmp_y_train, tmp_select_label_list)
                    elif params.extra_annotate_type=='current':
                        tmp_select_label_list = list(range(1+entity_index_min*(len(params.schema)-1), 1+(entity_idx_ed+1)*(len(params.schema)-1)))
                        tmp_y_train = self.set_unseen_labels_to_O(tmp_y_train, tmp_select_label_list)
                    elif params.extra_annotate_type=='all':
                        tmp_select_label_list = list(range(1+entity_index_min*(len(params.schema)-1), 1+(entity_index_max+1)*(len(params.schema)-1)))
                        tmp_y_train = self.set_unseen_labels_to_O(tmp_y_train, tmp_select_label_list)
                    else:
                        raise Exception('Invalid extra_annotate_type type %s!!!'%(params.extra_annotate_type))
                    y_train.extend(tmp_y_train)
                    entity_idx_bg = entity_idx_ed+1

            if not is_ground_truth_train and 'train' in phase:
                for i, y_lst in enumerate(y_train):
                    for j, y in enumerate(y_lst):
                        if y in [pad_token_label_id, self.label_list.index('O')]:
                            continue
                        if not y in select_label_index:
                            y_train[i][j] = self.label_list.index('O')
            if 'dev' in phase:
                for i, y_lst in enumerate(y_dev):
                    for j, y in enumerate(y_lst):
                        if y in [pad_token_label_id, self.label_list.index('O')]:
                            continue
                        if not y in select_label_index:
                            y_dev[i][j] = self.label_list.index('O')
            if 'test' in phase:
                for i, y_lst in enumerate(y_test):
                    for j, y in enumerate(y_lst):
                        if y in [pad_token_label_id, self.label_list.index('O')]:
                            continue
                        if not y in select_label_index:
                            y_test[i][j] = self.label_list.index('O')
        else:
            logger.info("Select all classes for training !")

        # Filter out all sentences contains ONLY O labels
        filter_label_list = []
        for filter_entity in filter_entity_list:
            if 'B-'+filter_entity in self.label_list:
                filter_label_list.append(self.label_list.index('B-'+filter_entity))
            if 'I-'+filter_entity in self.label_list:
                filter_label_list.append(self.label_list.index('I-'+filter_entity))
            if 'E-'+filter_entity in self.label_list:
                filter_label_list.append(self.label_list.index('E-'+filter_entity))
            if 'S-'+filter_entity in self.label_list:
                filter_label_list.append(self.label_list.index('S-'+filter_entity))

        if is_filter_O:
            logger.info('Filter out all sentences contains ONLY O labels...')
            if 'train' in phase:
                filter_O_inputs_train, filter_O_y_train = [], []
                for x_lst, y_lst in zip(inputs_train, y_train):
                    is_append=False
                    for y_label in y_lst:
                        if not (y_label in [-100, 0]+filter_label_list):
                            is_append=True
                            break
                    if is_append:
                        filter_O_inputs_train.append(x_lst)
                        filter_O_y_train.append(y_lst)
                retain_ratio_train = int(len(filter_O_y_train)/len(y_train)*100)
                logger.info('retain_ratio_train=%d%%'%(retain_ratio_train))
                inputs_train, y_train = filter_O_inputs_train, filter_O_y_train
            if 'dev' in phase:
                filter_O_inputs_dev, filter_O_y_dev = [], []
                for x_lst, y_lst in zip(inputs_dev, y_dev):
                    is_append=False
                    for y_label in y_lst:
                        if not (y_label in [-100, 0]+filter_label_list):
                            is_append=True
                            break
                    if is_append:
                        filter_O_inputs_dev.append(x_lst)
                        filter_O_y_dev.append(y_lst)
                retain_ratio_dev = int(len(filter_O_y_dev)/len(y_dev)*100)
                logger.info('retain_ratio_dev=%d%%'%(retain_ratio_dev))
                inputs_dev, y_dev = filter_O_inputs_dev, filter_O_y_dev
            if 'test' in phase:
                filter_O_inputs_test, filter_O_y_test = [], []
                for x_lst, y_lst in zip(inputs_test, y_test):
                    is_append=False
                    for y_label in y_lst:
                        if not (y_label in [-100, 0]+filter_label_list):
                            is_append=True
                            break
                    if is_append:
                        filter_O_inputs_test.append(x_lst)
                        filter_O_y_test.append(y_lst)
                retain_ratio_test = int(len(filter_O_y_test)/len(y_test)*100)
                logger.info('retain_ratio_test=%d%%'%(retain_ratio_test))
                inputs_test, y_test = filter_O_inputs_test, filter_O_y_test

        if 'train' in phase:
            dataset_train = Dataset(inputs_train, y_train)
            dataloader_train = DataLoader(dataset=dataset_train, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, # Don't shuffle, or the DCE went wrong
                                        collate_fn=collate_fn)
            return_result.append(dataloader_train)
        if 'dev' in phase:
            dataset_dev = Dataset(inputs_dev, y_dev)
            dataloader_dev = DataLoader(dataset=dataset_dev, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        collate_fn=collate_fn)
            return_result.append(dataloader_dev)
        if 'test' in phase:
            dataset_test = Dataset(inputs_test, y_test)
            dataloader_test = DataLoader(dataset=dataset_test, 
                                            batch_size=self.batch_size, 
                                            shuffle=False, 
                                            collate_fn=collate_fn)
            return_result.append(dataloader_test)
        
        return tuple(return_result)

if __name__ == "__main__":
    pass
    # spilt_dataset_random(['datasets/NER_data/conll2003'], 'train', domain2entity['conll2003'], 1, 1, 'BIO')
    # print_split_data_statistic(['datasets/NER_data/conll2003'], 'train', domain2entity['conll2003'], 1, 1, 'BIO')
    # get_entity_list('datasets/NER_data/i2b2/train.txt')
    # convert_BIOES_to_BIO('datasets/NER_data/ontonotes5/train.txt','datasets/NER_data/ontonotes5/train_.txt')
    # convert_BIOES_to_BIO('datasets/NER_data/ontonotes5/test.txt','datasets/NER_data/ontonotes5/test_.txt')
    # convert_BIOES_to_BIO('datasets/NER_data/ontonotes5/dev.txt','datasets/NER_data/ontonotes5/dev_.txt')