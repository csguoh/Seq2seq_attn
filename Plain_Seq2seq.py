from collections import Counter
from operator import mod
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk


class PlainEncoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout = 0.2):
        super(PlainEncoder,self).__init__()
        self.embed =nn.Embedding(vocab_size,hidden_size)

        self.rnn  = nn.GRU(hidden_size,hidden_size,batch_first = True)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x,lengths):
        """
        1 从长到短排序
        2 pack
        3 rnn
        4 unpack
        """
        sorted_len,sorted_idx = lengths.sort(0,descending = True)
        x_sorted  = x[sorted_idx.long()]
        enbeded = self.dropout(self.embed(x_sorted))
        packed_embeded = nn.utils.rnn.pack_padded_sequence(enbeded,sorted_len.long().cpu().data.numpy(),batch_first = True)
        packed_out, hid = self.rnn(packed_embeded)
        out,_ = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first =True)

        _,oringinal_idx = sorted_idx.sort(0,descending=False)
        out = out[oringinal_idx.long()].contiguous()
        hid = hid[:,oringinal_idx.long()].contiguous()

        return out, hid[[-1]]
    
    
class PlainDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout = 0.2):
        super(PlainDecoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,batch_first =True)
        self.out = nn.Linear(hidden_size,vocab_size)
        self.dropout =nn.Dropout(dropout)
    
    def forward(self,y,y_lengths,hid):

        sorted_len, sorted_idx = y_lengths.sort(0,descending = True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:,sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))

        packed_seq=nn.utils.rnn.pack_padded_sequence(y_sorted,sorted_len.long().cpu().data.numpy(),batch_first = True)

        out,hid = self.rnn(packed_seq,hid)
        unpacked,_= nn.utils.rnn.pad_packed_sequence(out,batch_first = True)

        _,original_idx = sorted_idx.sort(0,descending= False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:,original_idx.long()].contiguous()

        output = F.log_softmax(self.out(output_seq),-1)

        return output, hid


class PlainSeq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(PlainSeq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,x,x_lengths,y,y_lengths):
        encoder_out, hid = self.encoder(x,x_lengths)
        output,hid  = self.decoder(y,y_lengths,hid)

        return output,None
    
    def translate(self,x,x_lengths,y,max_length = 10):
        encoder_out,hid = self.encoder(x,x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for  i in range(max_length):
            output,hid = self.decoder(y,torch.ones(batch_size).long().to(y.device),
            hid)
            y = output.max(2)[1].view(batch_size,1)
            preds.append(y)
        
        return torch.cat(preds,1),None


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion,self).__init__()

    def forward(self,input,target,mask):
        input = input.contiguous().view(-1,input.size(2))
        target = target.contiguous().view(-1,1)
        mask = mask.contiguous().view(-1,1)
        output = -input.gather(1,target)*mask
        output = torch.sum(output) / torch.sum(mask)

        return output