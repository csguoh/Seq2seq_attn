from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk


class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hidden_size,dec_hidden_size,dropout = 0.2):
        super(Encoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.RNN = nn.GRU(embed_size,enc_hidden_size,batch_first = True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size*2, dec_hidden_size)
    
    def forward(self,x,lengths):
        sorted_len,sorted_idx = lengths.sort(0,descending=True)
        x_sorted = x[sorted_idx.long()]
        embeded = self.dropout(self.embed(x_sorted))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeded,sorted_len.long().cpu().data.numpy(),batch_first=True)
        packed_out,hid = self.RNN(packed_embedded)
        out,_ = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True)
        _,original_idx = sorted_idx.sort(0,descending = False)
        out = out[original_idx.long()].contiguous()
        hid=hid[:,original_idx.long()].contiguous()

        hid = torch.cat([hid[-2],hid[-1]],dim = 1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out,hid


class Attention(nn.Module):
    def __init__(self,enc_hidden_size,dec_hidden_size):
        super(Attention,self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2,dec_hidden_size,bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2+dec_hidden_size,dec_hidden_size)

    def forward(self,output,context,mask):
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size*input_len,-1)).view(batch_size,input_len,-1)

        attn = torch.bmm(output,context_in.transpose(1,2))

        attn.data.masked_fill(mask,-1e6)

        attn = F.softmax(attn,dim = 2)

        context = torch.bmm(attn,context)

        output = torch.cat((context,output),dim=2)
        output = output.view(batch_size*output_len,-1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size,output_len,-1)

        return output,attn


class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hidden_size,dec_hidden_size,dropout=0.2):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.atention = Attention(enc_hidden_size,dec_hidden_size)
        self.rnn = nn.GRU(embed_size,dec_hidden_size,batch_first= True)
        self.out = nn.Linear(dec_hidden_size,vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def creat_mask(self,x_len,y_len):
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        x_mask = x_mask.float()
        y_mask = y_mask.float()
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask
    
    def forward(self,ctx,ctx_lengths,y,y_lengths,hid):
        sorted_len,sorted_idx = y_lengths.sort(0,descending=True)
        y_sorted = y[sorted_idx.long()]
        hid =hid[:,sorted_idx.long()]
        y_sorted = self.dropout(self.embed(y_sorted))

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted,sorted_len.long().cpu().numpy(),batch_first=True)
        out,hid =self.rnn(packed_seq,hid)
        unpacked,_ = nn.utils.rnn.pad_packed_sequence(out,batch_first=True)

        _,origional_idx = sorted_idx.sort(0,descending=False)
        output_seq = unpacked[origional_idx.long()].contiguous()
        hid = hid[:,origional_idx.long()].contiguous()
        mask = self.creat_mask(y_lengths,ctx_lengths)
        output,attn  = self.atention(output_seq,ctx,mask)
        output = F.log_softmax(self.out(output),-1)

        return output,hid,attn
    

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder= decoder
    
    def forward(self,x,x_lengths,y,y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out, 
                    ctx_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    hid=hid)
        return output, attn
    
    def translate(self,x,x_lengths,y,max_lengths=100):
        encoder_out,hid = self.encoder(x,x_lengths)
        preds = []
        batch_size =x.shape[0]
        attns = []
        for i in range(max_lengths):
            output, hid, attn = self.decoder(ctx=encoder_out, 
                    ctx_lengths=x_lengths,
                    y=y,
                    y_lengths=torch.ones(batch_size).long().to(y.device),
                    hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)
