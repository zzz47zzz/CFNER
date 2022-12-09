import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import scipy
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score
from transformers import AutoTokenizer

from src.dataloader import *
from src.utils import *

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

class BaseTrainer(object):
    def __init__(self, params, model, entity_list, label_list):
        # parameters
        self.params = params
        self.model = model
        self.label_list = label_list
        self.entity_list = entity_list
        
        # training
        self.lr = float(params.lr)
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_acc = 0
        self.epoch = -1

        # init moving average
        self.embed_mean = torch.zeros(self.model.hidden_dim).numpy()
        self.mu = 0.9
        self.weight_decay = 5e-4

    
    def select_O_samples(self, refer_flatten_feat_O_train, O_pos_matrix, sample_strategy='all', sample_ratio=1.0, ground_truth_O_pos_matrix_list=[], old_class_list=[]):
        assert refer_flatten_feat_O_train.shape[0]==O_pos_matrix.shape[0],\
            "refer_flatten_feat_O_train.shape[0]!=O_pos_matrix.shape[0] !!!"
        assert self.refer_model!=None, "refer_model is none !!!"
        assert sample_ratio>0.0 and sample_ratio<=1.0, "invalid sample ratio %.4f!!!"%sample_ratio
        # get the logits of the refer model
        with torch.no_grad():
            self.refer_model.eval()
            O_logits = []
            for O_feature in refer_flatten_feat_O_train.split(64):
                O_feature = O_feature.cuda()
                O_logits.append(self.refer_model.classifier(O_feature).cpu())        
        O_logits = torch.cat(O_logits, dim=0)
        O_predicts = torch.argmax(O_logits, dim=-1)

        # Plot score distribution for curriculum learning
        # O_max_scores = np.array(torch.max(torch.softmax(O_logits,dim=-1), dim=-1)[0].cpu())
        # plt.hist(O_max_scores)
        # plt.show()
        
        if sample_strategy=='all':
            select_mask = torch.not_equal(O_predicts, self.label_list.index('O'))
        elif sample_strategy=='highest_prob':
            select_mask = torch.zeros_like(O_predicts)
            for label_id in list(range(O_logits.shape[-1])):
                if label_id==self.label_list.index('O'):
                    continue
                # find the samples has the top sample_ratio (default=0.1) predict probability
                num_label_sample = torch.sum(torch.eq(O_predicts, label_id))
                top_k_sample = int(num_label_sample*sample_ratio)
                top_k_pos = torch.topk(torch.softmax(O_logits,dim=-1)[:,label_id],
                                    k=top_k_sample,
                                    largest=True)[1] 
                # ensure the samples are predicted as label_id
                label_mask = torch.eq(O_predicts, label_id)
                top_k_mask = torch.zeros_like(label_mask)
                top_k_mask[top_k_pos] = 1
                # find the samples satisfy the above 2 requirements
                label_select_mask = torch.logical_and(label_mask, top_k_mask)
                
                select_mask = torch.logical_or(select_mask, label_select_mask)
        elif sample_strategy=='ground_truth':
            assert len(ground_truth_O_pos_matrix_list)>0, "ground_truth_O_pos_matrix is none!!!"
            assert len(old_class_list)>0, "old_class_list is none!!!"

            select_mask = torch.zeros(O_pos_matrix.shape[0]).bool()
            for old_class_id, ground_truth_O_pos_matrix in zip(old_class_list, ground_truth_O_pos_matrix_list):
                # ensure the samples are predicted as label_id
                label_mask = torch.eq(O_predicts, old_class_id)
                label_gt_mask = torch.zeros(O_pos_matrix.shape[0]).bool()
                for i, gt_O_pos in enumerate(ground_truth_O_pos_matrix):
                    i_pos, j_pos = gt_O_pos[0], gt_O_pos[1]
                    appear_mask = np.logical_and(\
                                            O_pos_matrix[:,0]==i_pos,
                                            O_pos_matrix[:,1]==j_pos)
                    appear_cnt = np.sum(appear_mask)
                    assert appear_cnt==1, "Find %d times of %s in O_pos_matrix!!!"%(appear_cnt, str(gt_O_pos))
                    label_gt_mask[np.where(appear_mask)[0].item()] = True
                
                label_select_mask = torch.logical_and(label_mask, label_gt_mask)
                select_mask = torch.logical_or(select_mask, label_select_mask)
                # assert torch.sum(label_gt_mask).item() == ground_truth_O_pos_matrix.shape[0]
                num_gt = len(ground_truth_O_pos_matrix)
                num_predict = torch.sum(label_mask).item()
                num_correct_predict = torch.sum(label_select_mask).item()
                if num_predict>0:
                    label_precision = num_correct_predict/num_predict
                else:
                    label_precision = 0
                if num_gt>0:
                    label_recall = num_correct_predict/num_gt
                else:
                    label_recall = 0
                if label_recall==0 or label_precision==0:
                    label_f1 = 0
                else:
                    label_f1 = 2/(1/label_precision+1/label_recall)*100
                logger.info('Label %d: precision=%.2f%% (%d/%d), recall=%.2f%% (%d/%d), f1=%.2f%%'%(
                    old_class_id, 
                    label_precision*100, num_correct_predict, num_predict,
                    label_recall*100, num_correct_predict, num_gt,
                    label_f1
                ))

        logger.info('Ratio of select samples %.2f%% (%d/%d).'%(\
                torch.sum(select_mask).item()/select_mask.shape[0]*100,
                torch.sum(select_mask).item(),
                select_mask.shape[0]
            )
        )

        return refer_flatten_feat_O_train[select_mask], O_pos_matrix[select_mask]

    def get_cl_prob_threshold(self, epoch):
        epoch_s, epoch_e = self.params.cl_epoch_start, self.params.cl_epoch_end
        pro_s, pro_e = self.params.cl_prob_start, self.params.cl_prob_end
        if epoch<epoch_s:
            return pro_s
        elif epoch>epoch_e:
            return pro_e
        else:
            return pro_s+(epoch-epoch_s)/(epoch_e-epoch_s)*(pro_e-pro_s)

    def get_cl_temperature(self, epoch):
        epoch_s, epoch_e = self.params.cl_epoch_start, self.params.cl_epoch_end
        tmp_s, tmp_e = self.params.cl_tmp_start, self.params.cl_tmp_end
        if epoch<epoch_s:
            return tmp_s
        elif epoch>epoch_e:
            return tmp_e
        else:
            return np.power(10,np.log10(tmp_s)+(epoch-epoch_s)/(epoch_e-epoch_s)*(np.log10(tmp_e)-np.log10(tmp_s)))

    def batch_forward(self, inputs, match_id_batch=None, O_match_id_batch=None, max_seq_length=512):    
        # Compute features
        self.inputs = inputs
        self.features = self.model.forward_encoder(inputs)
        # Compute logits
        self.logits = self.model.forward_classifier(self.features)   

        if match_id_batch!=None and len(match_id_batch)==0:
            self.features_match = None
            self.logits_match = None
        elif match_id_batch!=None and len(match_id_batch)>0:
            assert self.pad_token_id!=None, "pad_token_id is none!"
            assert self.dataloader_train!=None, "dataloader_train is none!"
            assert self.pos_matrix.any(), "pos_matrix is none!"
            # compute the sentences related to the match samples
            match_pos_matrix_batch = torch.tensor(self.pos_matrix[match_id_batch]).view(-1,2)
            select_sentence_idx = match_pos_matrix_batch[:,0]
            unique_sentence_idx = torch.unique(select_sentence_idx)
            select_to_unique_map = [list(unique_sentence_idx).index(i) for i in select_sentence_idx]
            select_sentence_batch = []
            for idx in unique_sentence_idx:
                select_sentence_batch.append(self.dataloader_train.dataset.X[idx])
            # pad the sentence batch
            length_lst = [len(s) for s in select_sentence_batch]
            max_length = max(length_lst)
            select_batch = torch.LongTensor(len(select_sentence_batch), max_length).fill_(self.pad_token_id)
            for i, s in enumerate(select_sentence_batch):
                select_batch[i,:len(s)] = torch.LongTensor(s)
            if select_batch.shape[1]>max_seq_length:
                select_batch = select_batch[:,:max_seq_length].clone()
            select_batch = select_batch.cuda()
            # compute match feature
            with torch.no_grad():
                self.model.eval()
                tmp_features_match_lst = []
                for _select_batch in select_batch.split(8):
                    tmp_features_match_lst.append(self.model.forward_encoder(_select_batch))
                tmp_features_match = torch.cat(tmp_features_match_lst, dim=0)
                features_match = torch.FloatTensor(len(match_id_batch),tmp_features_match.shape[-1])
                for i, pos in enumerate(match_pos_matrix_batch):
                    assert len(pos)==2, "pos type is invalid"
                    pos_i = pos[0].item()
                    pos_j = pos[1].item()
                    features_match[i] = tmp_features_match[select_to_unique_map[i]][pos_j]
                self.features_match = features_match.cuda()
                self.logits_match = self.model.forward_classifier(self.features_match)
                self.model.train()  

            # For visualization
            # print('------------------------DCE Matched------------------------')
            # for i,sent in enumerate(select_batch):
            #     print('Sent %d-th: %s'%(i, decode_sentence(sent, auto_tokenizer)))
            # for i, pos in enumerate(match_pos_matrix_batch):
            #     print('Entity %d-th: %s'%(
            #             i, 
            #             decode_word_from_sentence(
            #                 select_batch[select_to_unique_map[i]], 
            #                 pos[1], 
            #                 auto_tokenizer
            #             )
            #         ))              

        if O_match_id_batch!=None and len(O_match_id_batch)==0:
            self.O_features_match = None
            self.O_logits_match = None
        elif O_match_id_batch!=None and len(O_match_id_batch)>0:
            assert self.pad_token_id!=None, "pad_token_id is none!"
            assert self.dataloader_train!=None, "dataloader_train is none!"
            assert self.O_pos_matrix.any(), "O_pos_matrix is none!"
            # compute the sentences related to the match samples
            match_O_pos_matrix_batch = torch.tensor(self.O_pos_matrix[O_match_id_batch])
            select_sentence_idx = match_O_pos_matrix_batch[:,0]
            unique_sentence_idx = torch.unique(select_sentence_idx)
            select_to_unique_map = [list(unique_sentence_idx).index(i) for i in select_sentence_idx]
            select_sentence_batch = []
            for idx in unique_sentence_idx:
                select_sentence_batch.append(self.dataloader_train.dataset.X[idx])
            # pad the sentence batch
            length_lst = [len(s) for s in select_sentence_batch]
            max_length = max(length_lst)
            select_batch = torch.LongTensor(len(select_sentence_batch), max_length).fill_(self.pad_token_id)
            for i, s in enumerate(select_sentence_batch):
                select_batch[i,:len(s)] = torch.LongTensor(s)
            if select_batch.shape[1]>max_seq_length:
                select_batch = select_batch[:,:max_seq_length].clone()
            select_batch = select_batch.cuda()
            # compute match feature
            with torch.no_grad():
                self.model.eval()
                tmp_O_features_match_lst = []
                for _select_batch in select_batch.split(8):
                    tmp_O_features_match_lst.append(self.model.forward_encoder(_select_batch))
                tmp_O_features_match = torch.cat(tmp_O_features_match_lst, dim=0)
                O_features_match = torch.FloatTensor(len(O_match_id_batch),tmp_O_features_match.shape[-1])
                for i, pos in enumerate(match_O_pos_matrix_batch):
                    assert len(pos)==2, "pos type is invalid"
                    pos_i = pos[0].item()
                    pos_j = pos[1].item()
                    O_features_match[i] = tmp_O_features_match[select_to_unique_map[i]][pos_j]
                self.O_features_match = O_features_match.cuda()
                self.O_logits_match = self.model.forward_classifier(self.O_features_match)
                self.model.train()

            # # For visualization
            # print('------------------------ODCE Matched------------------------')
            # for i,sent in enumerate(select_batch):
            #     print('Sentence %d-th: %s'%(i, decode_sentence(sent, auto_tokenizer)))
            # for i, pos in enumerate(self.O_pos_matrix_batch):
            #     print('Entity %d-th: %s'%(
            #             i, 
            #             decode_word_from_sentence(inputs[pos[0]], pos[1], auto_tokenizer)
            #         ))
            # for i, pos in enumerate(match_O_pos_matrix_batch):
            #     print('Matched entity %d-th: %s'%(
            #             i, 
            #             decode_word_from_sentence(select_batch[select_to_unique_map[i]], pos[1], auto_tokenizer)
            #         )) 
    
    def compute_DCE(self, labels, ce_mask):
        '''
            DCE for labeled samples
        '''
        assert torch.sum(ce_mask.float()) == int(self.logits_match.shape[0]/params.top_k) \
                and self.logits_match.shape[0]%params.top_k == 0, \
                "length of ce_mask and the number of match samples are not equal!!!"
        # joint ce_loss
        logits_prob = F.softmax(self.logits.view(-1,self.logits.shape[-1]), dim=-1)
        logits_prob_match = F.softmax(self.logits_match, dim=-1)
        # print(logits_prob_match)
        logits_prob_match = torch.mean(logits_prob_match.reshape(-1, params.top_k, logits_prob_match.size(-1)), dim=1)
        # print(logits_prob_match)
        # print(logits_prob[ce_mask.flatten()])
        logits_prob_joint = (logits_prob[ce_mask.flatten()]+logits_prob_match)/2

        ce_loss = F.nll_loss(torch.log(logits_prob_joint+1e-10), labels[ce_mask])

        return ce_loss

    def compute_ODCE(self, refer_logits, distill_mask):
        '''
            DCE for O samples
        '''
        refer_dims = refer_logits.shape[-1]
        assert self.O_pos_matrix_batch.any(), "O_pos_matrix_batch is none"

        # get mask for defined O samples
        defined_O_mask = torch.zeros(refer_logits.shape[:2]).cuda()
        defined_O_mask[self.O_pos_matrix_batch[:,0], self.O_pos_matrix_batch[:,1]] = 1
        defined_O_mask = torch.logical_and(distill_mask, defined_O_mask)
        assert torch.sum(defined_O_mask.float()) == int(self.O_logits_match.shape[0]/params.top_k) \
                and self.O_logits_match.shape[0]%params.top_k == 0, \
                "length of defined_O_mask and the number of 'O' match samples are not equal!!!"
        
        # get average scores of the matched samples
        # truncate the refer_dims before softmax to mitigate the imblance 
        # between old and new classes 
        O_logits_prob_match = F.softmax(
                            self.O_logits_match[:,:refer_dims]/self.params.temperature, 
                            dim=-1)
        O_logits_prob_match = torch.mean(O_logits_prob_match.view(-1, params.top_k, O_logits_prob_match.shape[-1]), dim=1)
        # get scores of the original samples
        old_class_score_all = F.softmax(
                            self.logits/self.params.temperature,
                            dim=-1)[:,:,:refer_dims]
        joint_old_class_score_all = old_class_score_all.clone()

        # curriculum learning
        if params.is_curriculum_learning:
            # select the samples with highest prob
            assert self.epoch!=-1, "Epoch should be given for curriculum learning!!!"
            prob_threshold = self.get_cl_prob_threshold(self.epoch) 

            # Plot histgram
            # plt.hist(torch.max(old_class_score_all[defined_O_mask],dim=-1)[0].cpu().detach().numpy())
            # plt.show()

            curriculum_mask = torch.max(old_class_score_all[defined_O_mask],dim=-1)[0]>=prob_threshold
            defined_O_curricumlum_mask = defined_O_mask.clone()
            for i in range(defined_O_curricumlum_mask.shape[0]):
                for j in range(defined_O_curricumlum_mask.shape[1]):
                    if defined_O_curricumlum_mask[i][j] and torch.max(old_class_score_all[i][j])<prob_threshold:
                        defined_O_curricumlum_mask[i][j]=False
                        
            # Compute KL divergence of distributions
            # 1.log(joint_distribution)
            joint_old_class_score_all[defined_O_curricumlum_mask] = (old_class_score_all[defined_O_curricumlum_mask]+O_logits_prob_match[curriculum_mask])/2
            joint_old_class_score = torch.log(joint_old_class_score_all[distill_mask]+1e-10).view(-1, refer_dims)
            # 2.ref_distribution
            # Sharpen the effect of the define O samples
            cl_temperature = self.get_cl_temperature(self.epoch)
            refer_logits[defined_O_curricumlum_mask] /= cl_temperature
            undefined_O_mask = torch.logical_and(distill_mask,torch.logical_not(defined_O_curricumlum_mask))
            refer_logits[undefined_O_mask] /= self.params.ref_temperature
            ref_old_class_score = F.softmax(
                                refer_logits[distill_mask], 
                                dim=-1).view(-1, refer_dims)
            # KL divergence
            distill_loss = nn.KLDivLoss(reduction='batchmean')(joint_old_class_score, ref_old_class_score)
        else:
            # TODO: KLDiv or CE ?
            # Compute KL divergence of distributions
            # 1.log(joint_distribution)
            # print(O_logits_prob_match)
            # print(old_class_score_all[defined_O_mask])
            joint_old_class_score_all[defined_O_mask] = (old_class_score_all[defined_O_mask]+O_logits_prob_match)/2
            joint_old_class_score = torch.log(joint_old_class_score_all[distill_mask]+1e-10).view(-1, refer_dims)
            # 2.ref_distribution
            refer_logits[defined_O_mask] /= 1e-10 # Equals to applying CE to defined O samples, others is KLDivLoss
            ref_old_class_score = F.softmax(
                                refer_logits[distill_mask]/self.params.ref_temperature, 
                                dim=-1).view(-1, refer_dims)
            # KL divergence
            distill_loss = nn.KLDivLoss(reduction='batchmean')(joint_old_class_score, ref_old_class_score)
        
        return distill_loss

    def compute_CE(self, labels, ce_mask):
        '''
            Cross-Entropy Loss
        '''
        all_dims = self.logits.shape[-1]
        ce_loss = nn.CrossEntropyLoss()(self.logits[ce_mask].view(-1, all_dims),
                                labels[ce_mask].flatten().long())
        return ce_loss

    def compute_KLDiv(self, refer_logits, distill_mask):
        '''
            KLDivLoss
        '''
        refer_dims = refer_logits.shape[-1]

        # 1.log(distribution)
        old_class_score = F.log_softmax(
                            self.logits[distill_mask]/self.params.temperature,
                            dim=-1)[:,:refer_dims].view(-1, refer_dims)
        # 2.ref_distribution
        ref_old_class_score = F.softmax(
                            refer_logits[distill_mask]/self.params.ref_temperature, 
                            dim=-1).view(-1, refer_dims)

        distill_loss = nn.KLDivLoss(reduction='batchmean')(old_class_score, ref_old_class_score)

        return distill_loss

    def batch_loss(self, labels):
        '''
            Cross-Entropy Loss
        '''
        self.loss = 0
        assert self.logits!=None, "logits is none!"

        # classification loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long())
        self.loss = ce_loss
        return ce_loss.item() 

    def batch_loss_distill(self, labels):
        '''
            Cross-Entropy Loss + Distillation loss(KLDivLoss)
        '''
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim
            
        # Check input
        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  
        # assert_no_old_samples(labels, refer_dims, all_dims, pad_token_label_id)

        # (1) CE loss
        ce_mask = torch.logical_and(labels>=refer_dims,labels!=pad_token_label_id) 
        if torch.sum(ce_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).cuda()
        elif params.is_DCE and self.logits_match!=None:
            ce_loss = self.compute_DCE(labels, ce_mask)
        else:
            ce_loss = self.compute_CE(labels, ce_mask)

        # (2) Ditsillation loss
        with torch.no_grad():
            self.refer_model.eval()
            refer_features = self.refer_model.forward_encoder(self.inputs)
            refer_logits = self.refer_model.forward_classifier(refer_features)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
        
        distill_mask = torch.logical_and(labels==0,labels!=pad_token_label_id)
        if torch.sum(distill_mask.float())==0:
            distill_loss = torch.tensor(0., requires_grad=True).cuda()
        elif params.is_ODCE and self.O_logits_match!=None:
            distill_loss = self.compute_ODCE(refer_logits, distill_mask)
        else:   
            distill_loss = self.compute_KLDiv(refer_logits, distill_mask)

        if not params.adaptive_distill_weight:
            distill_weight = params.distill_weight
        elif params.adaptive_schedule=='root':
            distill_weight = params.distill_weight*np.power((refer_dims-1)/(all_dims-refer_dims),0.5)
        elif params.adaptive_schedule=='linear':
            distill_weight = params.distill_weight*np.power((refer_dims-1)/(all_dims-refer_dims),1)
        elif params.adaptive_schedule=='square':
            distill_weight = params.distill_weight*np.power((refer_dims-1)/(all_dims-refer_dims),2)
        else:
            raise Exception('Invalid %s'%(params.adaptive_schedule))

        # (3) Ranking Loss
        if params.is_ranking_loss:
            mr_mask = torch.logical_and(labels<refer_dims,
                                        labels!=pad_token_label_id).flatten()
            labels_masked = labels.flatten().long()[mr_mask].view(-1, 1)
            mr_logits = self.logits.view(-1, self.logits.shape[-1])[mr_mask]
            gt_scores = mr_logits.gather(1, labels_masked).repeat(1, params.lucir_K)
            max_novel_scores = mr_logits[:, refer_dims:].topk(params.lucir_K, dim=1)[0]

            count = gt_scores.size(0)
            if count > 0:
                mr_loss = nn.MarginRankingLoss(margin=params.lucir_mr_dist)(gt_scores.view(-1), \
                    max_novel_scores.view(-1), torch.ones(count*params.lucir_K).cuda()) * params.lucir_lw_mr
            else:
                mr_loss = torch.tensor(0., requires_grad=True).cuda()
        
        # weighted sum
        if params.is_ranking_loss:
            self.loss = ce_loss + distill_weight*distill_loss + params.ranking_weight*mr_loss
            return ce_loss.item(), distill_weight*distill_loss.item() + params.ranking_weight*mr_loss.item()
        else:
            self.loss = ce_loss + distill_weight*distill_loss
            return ce_loss.item(), distill_weight*distill_loss.item()

    def batch_loss_lucir(self, labels):
        '''
            Cross-Entropy Loss + Distillation Loss(CosineEmbeddingLoss) + MarginRankingLoss 
        '''
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim

        # Check input
        assert self.refer_model != None, "refer_model is none!"
        assert self.inputs != None, "inputs is none!"
        assert self.inputs.shape[:2] == labels.shape[:2], "inputs and labels are not matched!"
        assert self.logits != None, "logits is none!"
        # assert_no_old_samples(labels, refer_dims, all_dims, pad_token_label_id)

        # (1) CE loss
        ce_mask = torch.logical_and(labels>=refer_dims,labels!=pad_token_label_id)
        if torch.sum(ce_mask.float()) == 0:
            ce_loss = torch.tensor(0., requires_grad=True).cuda()
        elif params.is_DCE and self.logits_match!=None:
            ce_loss = self.compute_DCE(labels, ce_mask)
        else:
            ce_loss = self.compute_CE(labels, ce_mask)

        # (2) distill loss
        lw_distill = params.lucir_lw_distill*math.sqrt(refer_dims/(all_dims-refer_dims))
        distill_mask = torch.logical_and(labels == 0, labels!=pad_token_label_id)
        # compute refer_features from refer_model
        with torch.no_grad():
            self.refer_model.eval()
            refer_features = self.refer_model.forward_encoder(self.inputs)
        if torch.sum(distill_mask.float()) == 0:
            distill_loss = torch.tensor(0., requires_grad=True).cuda()
        else:
            distill_loss = lw_distill * nn.CosineEmbeddingLoss()(
                self.features[distill_mask].view(-1, self.model.hidden_dim),
                refer_features[distill_mask].view(-1, self.model.hidden_dim),
                torch.ones(distill_mask.nonzero().size(0)).cuda()
            )

        # (3) MR loss
        # 旧类别+O类别，无replay时只有O类别，同distill_mask
        mr_mask = torch.logical_and(labels<refer_dims,
                                    labels!=pad_token_label_id).flatten()
        labels_masked = labels.flatten().long()[mr_mask].view(-1, 1)
        mr_logits = self.logits.view(-1, self.logits.shape[-1])[mr_mask]
        gt_scores = mr_logits.gather(1, labels_masked).repeat(1, params.lucir_K)
        max_novel_scores = mr_logits[:, refer_dims:].topk(params.lucir_K, dim=1)[0]

        count = gt_scores.size(0)
        if count > 0:
            mr_loss = nn.MarginRankingLoss(margin=params.lucir_mr_dist)(gt_scores.view(-1), \
                max_novel_scores.view(-1), torch.ones(count*params.lucir_K).cuda()) * params.lucir_lw_mr
        else:
            mr_loss = torch.tensor(0., requires_grad=True).cuda()

        self.loss = ce_loss + distill_loss + mr_loss

        return ce_loss.item(), distill_loss.item()+mr_loss.item()

    def batch_loss_podnet(self, labels):
        '''
            NCA Loss/Cross-Entropy Loss + Distillation Loss(CosineEmbeddingLoss+L2_norm)
        '''
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim

        # Check input
        assert self.refer_model != None, "refer_model is none!"
        assert self.inputs != None, "inputs is none!"
        assert self.inputs.shape[:2] == labels.shape[:2], "inputs and labels are not matched!"
        assert self.logits != None, "logits is none!"
        # assert_no_old_samples(labels, refer_dims, all_dims, pad_token_label_id)

        # (1) NCA loss
        lsc_mask = torch.logical_and(labels >= refer_dims,
                                     labels != pad_token_label_id).flatten()

        if torch.sum(lsc_mask.float()) == 0:
            lsc_loss = torch.tensor(0., requires_grad=True).cuda()
        elif params.podnet_is_nca:
            similarities = self.logits.view(-1, all_dims)[lsc_mask]
            targets = labels.flatten().long()[lsc_mask]
            margins = torch.zeros_like(similarities)
            margins[torch.arange(margins.shape[0]), targets] = params.podnet_nca_margin
            similarities = params.podnet_nca_scale * (similarities - params.podnet_nca_margin)

            similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

            disable_pos = torch.zeros_like(similarities)
            disable_pos[torch.arange(len(similarities)),
                        targets] = similarities[
                            torch.arange(len(similarities)), targets]

            numerator = similarities[torch.arange(similarities.shape[0]),
                                     targets]
            denominator = similarities - disable_pos

            losses = numerator - torch.log(torch.exp(denominator).sum(-1))

            lsc_loss = torch.mean(-losses)
        else:
            lsc_loss = nn.CrossEntropyLoss()(self.logits.view(
                -1, all_dims)[lsc_mask], labels.flatten().long()[lsc_mask])

        # (2) distill loss
        distill_mask = torch.logical_and(labels==0,
                                         labels != pad_token_label_id)
        with torch.no_grad():
            self.model.eval()
            all_features = self.model.encoder(self.inputs)[1]
            self.refer_model.eval()
            refer_all_features = self.refer_model.encoder(self.inputs)[1]
            refer_features = refer_all_features[-1]
            self.model.train()

        lw_pod_flat = params.podnet_lw_pod_flat * math.sqrt(refer_dims/(all_dims-refer_dims))
        pod_flat_loss = lw_pod_flat * nn.CosineEmbeddingLoss(reduction='mean')(
                            self.features[distill_mask].view(-1, self.model.hidden_dim),
                            refer_features[distill_mask].view(-1, self.model.hidden_dim),
                            torch.ones(distill_mask.nonzero().size(0)).cuda())

        pod_spatial_loss = torch.tensor(0., requires_grad=True).cuda()
        for i, (a, b) in enumerate(zip(all_features, refer_all_features)):
            # shape of (batch_size, sent_len, hidden_dims)
            assert a.shape == b.shape, (a.shape, b.shape)

            a, b = a[distill_mask], b[distill_mask]

            if params.podnet_normalize:
                a = F.normalize(a, dim=-1, p=2)
                b = F.normalize(b, dim=-1, p=2)

            a = a.sum(dim=-1).unsqueeze(-1)# (-1, 1)
            b = b.sum(dim=-1).unsqueeze(-1)# (-1, 1)

            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
            pod_spatial_loss += layer_loss

        pod_spatial_loss = params.podnet_lw_pod_spat * (pod_spatial_loss/len(all_features))

        self.loss = lsc_loss + pod_flat_loss + pod_spatial_loss

        return lsc_loss.item(), pod_flat_loss.item() + pod_spatial_loss.item()
            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()        
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

    def evaluate(self, dataloader, each_class=False, entity_order=[], is_plot_hist=False, is_save_txt=False, is_plot_cm=False):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []

            for x, y in dataloader: 
                x, y = x.cuda(), y.cuda()
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu()
                x_list.append(x) 
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)
            
            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)   
            pred_list = torch.argmax(logits_list, dim=-1)

            ### Plot the (logits) prob distribution for each class
            if is_plot_hist:
                plot_prob_hist_each_class(deepcopy(y_list), 
                                        deepcopy(logits_list),
                                        ignore_label_lst=[
                                            self.label_list.index('O'),
                                            pad_token_label_id
                                        ])

            ### save the txt file
            if is_save_txt:
                save_predicts_to_txt(deepcopy(x_list),
                                deepcopy(y_list),
                                deepcopy(pred_list), 
                                label_list=self.label_list,
                                pred_file_name='pred_file.txt',
                                pad_token_label_id=pad_token_label_id,
                                model_name=self.params.model_name)

            ### for confusion matrix visualization
            if is_plot_cm:
                plot_confusion_matrix(deepcopy(pred_list),
                                deepcopy(y_list), 
                                label_list=self.label_list,
                                pad_token_label_id=pad_token_label_id)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            for pred_index, gold_index in zip(pred_list, y_list):
                gold_index = int(gold_index)
                if gold_index != pad_token_label_id:
                    pred_token = self.label_list[pred_index]
                    gold_token = self.label_list[gold_index]
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    pred_line.append(pred_token) 
                    gold_line.append(gold_token) 

            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            self.model.train()

            # compute overall f1 score
            # micro f1 (default)
            f1 = f1_score([gold_line], [pred_line])*100
            # macro f1 (average of each class f1)
            ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
            if not each_class:
                return f1, ma_f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, ma_f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ma_f1, ordered_f1_score_dict

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "encoder": self.model.encoder.state_dict(),
            "classifier": self.model.classifier
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)

        self.model.hidden_dim = ckpt['hidden_dim']
        self.model.output_dim = ckpt['output_dim']
        self.model.encoder.load_state_dict(ckpt['encoder'])
        self.model.classifier = ckpt['classifier']
        logger.info("Model has been load from %s" % load_path)