import math
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from transformers import AutoConfig
from transformers import AutoModelWithLMHead

from src.utils import *
from src.dataloader import *

logger = logging.getLogger()

class BertTagger(nn.Module):

    def __init__(self, output_dim, params):
        super(BertTagger, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.output_dim = output_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        self.encoder = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        if params.ckpt:
            logger.info("Reloading encoder from %s" % params.ckpt)
            encoder_ckpt = torch.load(params.ckpt)
            self.encoder.load_state_dict(encoder_ckpt)
        self.classifier = CosineLinear(self.hidden_dim, self.output_dim)
        # self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        # self.classifier = Causal_Norm_Classifier(self.output_dim, 
        #                                         self.hidden_dim, 
        #                                         alpha=params.alpha)

    def forward(self, X, return_feat=False):
        features = self.forward_encoder(X)
        logits = self.forward_classifier(features)
        if return_feat:
            return logits, features
        return logits
        
    def forward_encoder(self, X):
        features = self.encoder(X) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        features = features[1][-1] # (bsz, seq_len, hidden_dim)
        return features

    def forward_classifier(self, features, embed_mean=None):
        logits = self.classifier(features)
        # logits = self.classifier(features, embed_mean)
        return logits

class Causal_Norm_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, num_head=2, tau=16.0, alpha=0, gamma=0.03125, *args):
        super(Causal_Norm_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.reset_parameters(self.weight)
        self.embed_mean = None
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, embed_mean=None):

        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x.view(-1,x.shape[-1]))
        y = torch.mm(normed_x * self.scale, normed_w.t())

        # update the mean embedding
        if not embed_mean is None:
            self.embed_mean = embed_mean

        # remove the effect of confounder c during test
        if not self.training:

            _embed_mean = torch.from_numpy(self.embed_mean).view(1, -1).to(x.device)
            normed_c = self.multi_head_call(self.l2_norm, _embed_mean)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y_origin = torch.mm(nx * self.scale, nw.t())
                y_bias = torch.mm(cos_val * self.alpha * nc * self.scale, nw.t())
                y0 = y_origin - y_bias
                output.append(y0)
            y = sum(output)
            
        return y

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x

class CosineLinear(nn.Module):
    def __init__(self, hidden_dim, output_dim, sigma=True):
        super(CosineLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(output_dim, hidden_dim))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input, num_head=1):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        if num_head>1:
            out=[]
            head_dim = input.size(1)//num_head
            input_list = torch.split(input, head_dim, dim=1)
            input_list = [F.normalize(input_item, p=2,dim=1) for input_item in input_list]
            weight_list = torch.split(self.weight, head_dim, dim=1)
            weight_list = [F.normalize(weight_item, p=2,dim=1) for weight_item in weight_list]
            for n_input, n_weight in zip(input_list, weight_list):
                out.append(F.linear(n_input, n_weight))
            import pdb; pdb.set_trace()
            out = sum(out)
        else:
            out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out

        return out

class SplitCosineLinear(nn.Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, hidden_dim, old_output_dim, new_output_dim, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = old_output_dim + new_output_dim
        self.fc0 = CosineLinear(hidden_dim, 1, False) # for "O" class
        self.fc1 = CosineLinear(hidden_dim, old_output_dim-1, False)
        self.fc2 = CosineLinear(hidden_dim, new_output_dim, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x, num_head=1):
        out0 = self.fc0(x, num_head=num_head)
        out1 = self.fc1(x, num_head=num_head)
        out2 = self.fc2(x, num_head=num_head)
        out = torch.cat((out0, out1, out2), dim=-1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out

