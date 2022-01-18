from collections import Counter
from operator import mod
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from Plain_Seq2seq import PlainEncoder,PlainDecoder,PlainSeq2Seq,LanguageModelCriterion
from Attention import Encoder,Decoder,Seq2Seq


def load_data(in_file):
    cn = []
    en = []
    num_exaples = 0
    with open(in_file,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            en.append(["BOS"]+nltk.word_tokenize(line[0].lower())+["EOS"])
            cn.append(["BOS"]+[c for c in line[1]]+['EOS'])
    
    return en,cn


def build_dict(sentences, max_words = 50000):
    word_count = Counter()
    for senrence in sentences:
        for s in senrence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    print(f"total words:{len(ls)+2}")

    word_dict = {w[0]:index+2 for index,w in enumerate(ls)}

    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] =PAD_IDX
    return word_dict, len(ls)+2


def encode(en_sentences, cn_sentences,en_dict,cn_dict,sorted_by_len = True):
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w,0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w,0) for w in sent] for sent in cn_sentences]
    def len_argsort(seq):
        return sorted(range(len(seq)), key= lambda x : len(seq[x]))
    
    if sorted_by_len :
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in  sorted_index]
        
    return out_en_sentences, out_cn_sentences


def get_minibatches(n,minibatch_size,shuffle = True):
    idx_list = np.arange(0,n,minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx+minibatch_size,n)))
    
    return minibatches


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples= len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples,max_len)).astype("int32")

    x_lengths = np.array(lengths).astype("int32")

    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    
    return x, x_lengths


def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]     
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
        
    return all_ex


def translate_dev(i):
    en_sent = ' '.join([inv_en_dict[w] for w in dev_en[i]])
    print('='*16)
    print(en_sent)
    cn_sent = ' '.join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))

    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)    
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
 
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS": # 把数值变成单词形式
            trans.append(word) #
        else:
            break
    print("".join(trans))


def train(model,data,num_epochs = 2):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.0
        for it,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data,start=1):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()

            mb_input = torch.from_numpy(mb_y[:,:-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:,1:]).to(device).long()

            mb_y_len =torch.from_numpy(mb_y_len-1).to(device).long()

            mb_y_len[mb_y_len<=0]=1

            mb_pred,attn = model(mb_x,mb_x_len,mb_input,mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(),device=device)[None,:]<mb_y_len[:,None]

            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred,mb_output,mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item()*num_words

            total_num_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if it % 100 == 0:
                print(f'Epoch:{epoch} iterations:{it} loss:{loss.item()} ')
                       
        print("Epoch", epoch, "Training loss", total_loss/total_num_words)


def evaluate(model,data):
    model.eval()
    total_num_words = total_loss = 0.0
    with torch.no_grad():
        for it,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)


train_file = "./data/nmt/en-cn/train.txt"
dev_file = "./data/nmt/en-cn/dev.txt"
dev_en,dev_cn = load_data(dev_file)
train_en,train_cn = load_data(train_file)

# print(train_en[:10])
# print(train_cn[:10])

UNK_IDX = 0
PAD_IDX = 1

en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words =build_dict(train_cn)

inv_en_dict = {v: k for k,v in en_dict.items()}
inv_cn_dict = {v: k for k, v in cn_dict.items()}


train_en,train_cn = encode(train_en,train_cn,en_dict,cn_dict) 
dev_en,dev_cn = encode(dev_en,dev_cn,en_dict,cn_dict)


batch_size = 64
train_data = gen_examples(train_en, train_cn, batch_size)
random.shuffle(train_data)
dev_data = gen_examples(dev_en, dev_cn, batch_size)

# print(train_data[0])

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dropout = 0.2
embed_size = hidden_size = 100
encoder = Encoder(vocab_size=en_total_words,
                       embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                       dec_hidden_size=hidden_size,
                      dropout=dropout)
decoder = Decoder(vocab_size=cn_total_words,
                      embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                       dec_hidden_size=hidden_size,
                      dropout=dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())

# train(model, train_data, num_epochs=80)
# torch.save(model.state_dict(),'./data/nmt/Seq2seq_attn.pkl')
model.load_state_dict(torch.load('./data/nmt/Seq2seq_attn.pkl'))

for i in range(100,120):
    translate_dev(i)
    print()