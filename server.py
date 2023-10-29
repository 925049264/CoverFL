
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3,4,5'
import os

import torch.nn.functional as F
import torch
import torch.utils.data as data
import re
import torch.nn as nn
import json
from torch import optim

from tqdm import tqdm
import numpy as np
import pickle
import random
import sys
from torch_geometric.nn import GATConv
from collections import Counter
from itertools import chain

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'SentenceLen': 10,
    'batch_size': 60,
    'embedding_size': 16,
    'WoLen': 15,
    'Vocsize': 100,
    'Nl_Vocsize': 100,
    'max_step': 3,
    'margin': 0.5,
    'poolsize': 50,
    'Code_Vocsize': 100,
    'seed': 0,
    'lr': 1e-3
})

class GraphLoss(torch.nn.Module):
    def __init__(self,num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(GraphLoss, self).__init__()
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.batched_semi_loss(h1, h2, batch_size)
        l1 = l1.mean() if mean else l1.sum()


        return l1

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            refl_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1)))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        #拉远负样本对应节点距离，拉近正样本对应节点距离，拉近视图内节点距离
#         device = z1.device
        num_nodes = z1.size(0)
        num_nod = int(num_nodes/batch_size)
        num_batches = int(batch_size/2) - 1
        f = lambda x: torch.exp(x / self.tau)
        losses = []
        z1_mask = z1[:num_nod,:]
        z1_full = z1[num_nod:,:]
        refl_sim = f(self.sim(z1_mask, z1_full))  # [B, N]
        between_sim = f(self.sim(z1_mask, z2))  # [B, N]
        self_sim = f(self.sim(z1_mask, z1_mask))
        for i in range(num_batches):
            losses.append(-torch.log(
                refl_sim[:, i * num_nod:(i + 1) * num_nod].diag()
                / between_sim[:, i * num_nod:(i + 1) * num_nod].diag()+refl_sim[:, i * num_nod:(i + 1) * num_nod].diag()+self_sim.sum()-self_sim.diag()))

        return torch.cat(losses)
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        # self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        '''self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3'''
        self.word2id["<pad>"] = 0
        # self.word2id["NothingHere"] = 0
        self.word2id["Unknown"] = 1
        '''self.word2id["Unknown"] = 1
        self.word2id["NothingHere"] = 0
        self.word2id["NoneCopy"] = 2
        self.word2id["CopyNode"] = 3
        self.word2id["<StartNode>"] = 4'''

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=0):
        vocab_entry = VocabEntry()
        # print(list(chain(*corpus)))
        word_freq = Counter(chain(*corpus))
        # print(word_freq)
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                       len(non_singletons)))
        print('singletons: %s' % singletons)

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]
        words_not_included = []
        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)
                else:
                    words_not_included.append(word)

        print('word types not included: %s' % words_not_included)

        return vocab_entry


class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))

dmap = {
    'Math': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
             15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29,
             28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42,
             41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51, 50: 52, 51: 53, 52: 54, 53: 55,
             54: 56, 55: 57, 56: 58, 57: 59, 58: 60, 59: 61, 60: 62, 61: 63, 62: 64, 63: 65, 64: 66, 65: 67, 66: 68,
             67: 69, 68: 70, 69: 71, 70: 72, 71: 73, 72: 74, 73: 75, 74: 76, 75: 77, 76: 78, 77: 79, 78: 80, 79: 81,
             80: 82, 81: 83, 82: 84, 83: 85, 84: 86, 85: 87, 86: 88, 87: 89, 88: 90, 89: 91, 90: 92, 91: 93, 92: 94,
             93: 95, 94: 96, 95: 97, 96: 98, 97: 99, 98: 100, 99: 101, 100: 102, 101: 103, 102: 105, 103: 106},
    'Lang': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
             15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30,
             28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43,
             41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57,
             54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    'Chart': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
              15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 25, 24: 26},
    'Time': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
             15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27},
    'Mockito': {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
                15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 29, 27: 30,
                28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38}
}


class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", proj="Math", testid=0, lst=[],eb_size = args.embedding_size):
        self.token_embedding = torch.nn.Embedding(32, eb_size-1)
        self.token_embedding1 = torch.nn.Embedding(32, eb_size)
        self.train_path = "data/"+ proj + ".pkl"
        self.val_path = "ndev.txt"  # "validD.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len(self.Nl_Voc)
        self.Nl_Voc['Test'] = len(self.Nl_Voc)
        self.Nl_Voc['Line'] = len(self.Nl_Voc)
        self.Nl_Voc['RTest'] = len(self.Nl_Voc)
#         self.Nl_Len = config.NlLen
#         self.Code_Len = config.CodeLen
#         self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        if os.path.exists("data/" +"nl_voc.pkl"):
            #    self.init_dic()
            self.Load_Voc()
        else:
            self.init_dic()
        print(self.Nl_Voc)
        if not os.path.exists("data/" +self.proj + 'data.pkl'):
            data = self.preProcessData(open(self.train_path, "rb"))
        else:
            data = pickle.load(open("data/" +self.proj + 'data.pkl', 'rb'))
        self.data = []
        if dataName == "train":
            for i in range(len(data)):
                # if testid == 0:
                #    self.data.append(data[i][testid + 1:])
                # elif testid == len(data[i]) - 1:
                #    self.data.append(data[i][0:testid] + data[i][testid + 1:])
                # else:
                tmp = []
                for j in range(len(data[i])):
                    if j in lst:
                        continue
                    tmp.append(data[i][j])  # self.data.append(data[i][0:testid] + data[i][testid + 1:])
                self.data.append(tmp)
            # for i in range(len(data)):
            #    self.data = data[0:testid] + data[testid + 1:]
        elif dataName == 'test':
            # self.data = self.preProcessData(open('Lang.pkl', 'rb'))
            # self.ids = []
            testnum = int(0.05 * len(data[0]))
            # print("testnum: ",testnum)
            ids = []
            while len(ids) < testnum:
                rid = random.randint(0, len(data[0]) - 1)
                if rid == testid or rid in ids or rid == 51:  # if rid >= testid * testnum and rid < testid * testnum + testnum or rid in ids:
                    # print("rid: ",rid)
                    continue
                ids.append(rid)
            self.ids = ids
            # print("ids: ",ids)
            for i in range(len(data)):
                tmp = []
                for x in self.ids:
                    tmp.append(data[i][x])
                self.data.append(tmp)
        else:
            testnum = 1  # int(0.1 * len(data[0]))
            ids = []
            for i in range(len(data)):
                tmp = []
                for x in range(testnum * testid, testnum * testid + testnum):
                    if x < len(data[i]):
                        if i == 0:
                            ids.append(x)
                        tmp.append(data[i][x])
                self.data.append(tmp)
            self.ids = ids

    def Load_Voc(self):
        if os.path.exists("data/" +"nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("data/" +"nl_voc.pkl", "rb"))
        if os.path.exists("data/" +"code_voc.pkl"):
            self.Code_Voc = pickle.load(open("data/" +"code_voc.pkl", "rb"))
        if os.path.exists("data/" +"char_voc.pkl"):
            self.Char_Voc = pickle.load(open("data/" +"char_voc.pkl", "rb"))

    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans

    def init_dic(self):
        print("initVoc")
        print("data/" +self.proj + '.pkl')
        f = open("data/" +self.proj+ '.pkl', 'rb')
        data = pickle.load(f)
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index('(')]
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
                print(Codes[-1])
            for s in x['ftest']:
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff=0)
        self.Code_Voc = code_voc.word2id
        open("data/" +"code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_w_EM(self, word, voc):
        return voc[word]

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans

    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_list(self, seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def tokenize_for_bleu_eval(self, code):
        code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
        # code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens

    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans

    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index(x) < self.Code_Len and x != "(" and x != ")":
                ans.append(len(self.Nl_Voc) + codetoken.index(x))
            else:
                if x in self.Nl_Voc:
                    ans.append(self.Nl_Voc[x])
                else:
                    ans.append(1)
        for x in ans:
            if x >= len(self.Nl_Voc) + self.Code_Len:
                print(codetoken, nltoken)
                exit(0)
        return ans

    def preProcessData(self, dataFile):
        path_stacktrace = os.path.join('../FLocalization/stacktrace', self.proj)
        lines = pickle.load(dataFile)  # dataFile.readlines()
        Nodes = []
        rMask = []
        al_All_len = []
        LineNodes = []
        Res = []
        inputText = []
        inputNlad = []
        maxl = 0
        maxl2 = 0
        error = 0
        error1 = 0
        error2 = 0
        correct = 0

        for k in range(len(lines)):
            x = lines[k]
            if not isinstance(x['rtest'], dict):

                print(k)
                nodes = []
                res = []
                nladrow = []  # = np.zeros([3200, 3200])
                nladcol = []
                nladval = []
                texta = []
                textb = []
                linenodes = []
                methodnum = len(x['methods'])
                # print("x['ftest']: ", x['ftest'])
                # print("x['rtest']: ", x['rtest'])
                all_lenth = len(x['methods']) + len(x['ftest']) + int(x['rtest'])

                rrdict = {}
                for s in x['methods']:
                    rrdict[x['methods'][s]] = s[:s.index('(')]
                for i in range(methodnum):
                    nodes.append('Method')
                    if len(rrdict[i].split(":")) > 1:
                        tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:] + [rrdict[i].split(":")[1]])
                    else:
                        tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:])
                    ans = self.splitCamel(tokens)
                    ans.remove('.')
                    texta.append(ans)
                    if i in x['ans']:
                        res.append(1)
                    else:
                        res.append(0)
                rrdic = {}
                for s in x['ftest']:
                    rrdic[x['ftest'][s]] = s
                for i in range(len(x['ftest'])):
                    nodes.append('Test')
                    if len(rrdic[i].split(":")) > 1:
                        tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:] + [rrdic[i].split(":")[1]])
                    else:
                        tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:])
                    ans = self.splitCamel(tokens)
                    ans.remove('.')
                    textb.append(ans)
                for i in range(x['rtest']):
                    nodes.append('RTest')

                for i in range(len(x['lines'])):
                    if i not in x['ltype']:
                        x['ltype'][i] = 'Empty'
                    if x['ltype'][i] not in self.Nl_Voc:
                        self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                    linenodes.append(x['ltype'][i])
                    if x['ltype'][i] == 2:
                        print(1111)
                '''for i in range(len(x['mutation'])):
                    if x['mutation'][i] not in self.Nl_Voc:
                        self.Nl_Voc[x['mutation'][i]] = len(self.Nl_Voc)
                    nodes.append(x['mutation'][i])
                    types.append(0)'''
                maxl = max(maxl, len(nodes))
                maxl2 = max(maxl2, len(linenodes))
                ed = {}
                all_len = len(nodes) + len(linenodes)
                line2method = {}
                for e in x['edge2']:
                    line2method[e[1]] = e[0]
                    a = e[0]
                    b = e[1] + all_lenth  # len(x['ftest']) + methodnum
                    # nlad[a, b] = 1
                    # nlad[b, a] = 1
                    # assert(0)
                    if (a, b) not in ed:
                        ed[(a, b)] = 1
                    else:
                        print(a, b)
                        assert (0)
                    if (b, a) not in ed:
                        ed[(b, a)] = 1
                    else:
                        print(a, b)
                        assert (0)
                    nladrow.append(a)
                    nladcol.append(b)
                    nladval.append(1)
                    nladrow.append(b)
                    nladcol.append(a)
                    nladval.append(1)
                for e in x['edge10']:
                    if e[0] not in line2method:
                        error1 += 1
                    a = e[0] + all_lenth
                    b = e[1] + methodnum + len(x['ftest'])
                    nladrow.append(a)
                    nladcol.append(b)
                    if (a, b) not in ed:
                        ed[(a, b)] = 1
                    else:
                        pass
                        # print(e[0])
                        # print(a, b)
                        # assert(0)
                    if (b, a) not in ed:
                        ed[(b, a)] = 1
                    else:
                        pass

                        # print(a, b)
                        # assert(0)
                    nladval.append(1)
                    nladrow.append(b)
                    nladcol.append(a)
                    nladval.append(1)
                for e in x['edge']:
                    if e[0] not in line2method:
                        error2 += 1
                    a = e[0] + all_lenth  # + len(x['ftest']) + methodnum
                    b = e[1] + methodnum
                    nladrow.append(a)
                    nladcol.append(b)
                    if (a, b) not in ed:
                        ed[(a, b)] = 1
                    else:
                        print(e[0])
                        print(a, b)
                        assert (0)
                    if (b, a) not in ed:
                        ed[(b, a)] = 1
                    else:
                        print(a, b)
                        assert (0)
                    nladval.append(1)
                    nladrow.append(b)
                    nladcol.append(a)
                    nladval.append(1)
                    # nlad[a, b] = 1
                    # nlad[b, a] = 1

                overlap = self.getoverlap(texta, textb)
                overlap = self.pad_seq(overlap, len(nodes))
                # print(overlap)
                # Nodes.append(self.Get_Em(nodes, self.Nl_Voc))#Method、ptest、ftest
                # Res.append(self.pad_seq(res, all_len))#标签，标记method的对错
                # inputText.append(self.pad_seq(overlap, all_lenth))
                # LineNodes.append(self.Get_Em(linenodes, self.Nl_Voc))
                nd1 = torch.Tensor(self.Get_Em(nodes, self.Nl_Voc)).long()
                ld1 = torch.Tensor(self.Get_Em(linenodes, self.Nl_Voc)).long()
                resmask = torch.eq(torch.cat([nd1, ld1]), 2)
                nodes1 = self.token_embedding(nd1)
                linenodes1 = self.token_embedding1(ld1)
                nodeem = torch.cat([nodes1, torch.Tensor(overlap).unsqueeze(-1).float()], dim=-1)
                # print(nodeem.shape)
                # print(linenodes1.shape)
                xx = torch.cat([nodeem, linenodes1], dim=0)

                Nodes.append(xx.tolist())
                rMask.append(resmask.tolist())
                Res.append(self.pad_seq(res, all_len))
                # inputText.append(self.pad_seq(overlap, self.Nl_Len))
                # LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len))
                l_rtest = x['rtest']
                l_method = len(x['methods'])
                l_ftest = len(x['ftest'])
                lis1 = list()
                lis1.append(nladrow)
                lis1.append(nladcol)
                inputNlad.append(lis1)
                All_len = []
                All_len.append(all_len)
                All_len.append(l_method)
                All_len.append(l_ftest)
                All_len.append(l_rtest)
                al_All_len.append(All_len)

            else:
                nodes = []
                res = []
                nladrow = []  # = np.zeros([3200, 3200])
                nladcol = []
                nladval = []
                texta = []
                textb = []
                linenodes = []
                methodnum = len(x['methods'])
                # print("x['ftest']: ",x['ftest'])
                # print("x['rtest']: ",x['rtest'])
                all_lenth = len(x['methods']) +len(x['ftest'] ) +len(x['rtest'])

                rrdict = {}
                for s in x['methods']:
                    rrdict[x['methods'][s]] = s[:s.index('(')]
                for i in range(methodnum):
                    nodes.append('Method')
                    if len(rrdict[i].split(":")) > 1:
                        tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:] + [rrdict[i].split(":")[1]])
                    else:
                        tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:])
                    ans = self.splitCamel(tokens)
                    ans.remove('.')
                    texta.append(ans)
                    if i in x['ans']:
                        res.append(1)
                    else:
                        res.append(0)
                rrdic = {}
                for s in x['ftest']:
                    rrdic[x['ftest'][s]] = s
                for i in range(len(x['ftest'])):
                    nodes.append('Test')
                    if len(rrdic[i].split(":")) > 1:
                        tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:] + [rrdic[i].split(":")[1]])
                    else:
                        tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:])
                    ans = self.splitCamel(tokens)
                    ans.remove('.')
                    textb.append(ans)
                for i in range(len(x['rtest'])):
                    nodes.append('RTest')

                for i in range(len(x['lines'])):
                    if i not in x['ltype']:
                        x['ltype'][i] = 'Empty'
                    if x['ltype'][i] not in self.Nl_Voc:
                        self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                    linenodes.append(x['ltype'][i])
                    if x['ltype'][i] == 2:
                        print(1111)
                '''for i in range(len(x['mutation'])):
                    if x['mutation'][i] not in self.Nl_Voc:
                        self.Nl_Voc[x['mutation'][i]] = len(self.Nl_Voc)
                    nodes.append(x['mutation'][i])
                    types.append(0)'''
                maxl = max(maxl, len(nodes))
                maxl2 = max(maxl2, len(linenodes))
                ed = {}
                all_len = len(nodes ) +len(linenodes)
                line2method = {}
                for e in x['edge2']:
                    line2method[e[1]] = e[0]
                    a = e[0]
                    b = e[1] + all_lenth  # len(x['ftest']) + methodnum
                    # nlad[a, b] = 1
                    # nlad[b, a] = 1
                    # assert(0)
                    if (a, b) not in ed:
                        ed[(a, b)] = 1
                    else:
                        print(a, b)
                        assert (0)
                    if (b, a) not in ed:
                        ed[(b, a)] = 1
                    else:
                        print(a, b)
                        assert (0)
                    nladrow.append(a)
                    nladcol.append(b)
                    nladval.append(1)
                    nladrow.append(b)
                    nladcol.append(a)
                    nladval.append(1)
                for e in x['edge10']:
                    if e[0] not in line2method:
                        error1 += 1
                    a = e[0] + all_lenth
                    b = e[1] + methodnum + len(x['ftest'])
                    nladrow.append(a)
                    nladcol.append(b)
                    if (a, b) not in ed:
                        ed[(a, b)] = 1
                    else:
                        pass
                        # print(e[0])
                        # print(a, b)
                        # assert(0)
                    if (b, a) not in ed:
                        ed[(b, a)] = 1
                    else:
                        pass

                        # print(a, b)
                        # assert(0)
                    nladval.append(1)
                    nladrow.append(b)
                    nladcol.append(a)
                    nladval.append(1)
                for e in x['edge']:
                    if e[0] not in line2method:
                        error2 += 1
                    a = e[0] + all_lenth # + len(x['ftest']) + methodnum
                    b = e[1] + methodnum
                    nladrow.append(a)
                    nladcol.append(b)
                    if (a, b) not in ed:
                        ed[(a, b)] = 1
                    else:
                        print(e[0])
                        print(a, b)
                        assert (0)
                    if (b, a) not in ed:
                        ed[(b, a)] = 1
                    else:
                        print(a, b)
                        assert (0)
                    nladval.append(1)
                    nladrow.append(b)
                    nladcol.append(a)
                    nladval.append(1)
                    # nlad[a, b] = 1
                    # nlad[b, a] = 1

                overlap = self.getoverlap(texta, textb)
                overlap = self.pad_seq(overlap, len(nodes))
                # print(overlap)
                # Nodes.append(self.Get_Em(nodes, self.Nl_Voc))#Method、ptest、ftest
                # Res.append(self.pad_seq(res, all_len))#标签，标记method的对错
                # inputText.append(self.pad_seq(overlap, all_lenth))
                # LineNodes.append(self.Get_Em(linenodes, self.Nl_Voc))
                nd1 = torch.Tensor(self.Get_Em(nodes, self.Nl_Voc)).long()
                ld1 = torch.Tensor(self.Get_Em(linenodes, self.Nl_Voc)).long()
                resmask = torch.eq(torch.cat([nd1 ,ld1]), 2)
                nodes1 = self.token_embedding(nd1)
                linenodes1 = self.token_embedding1(ld1)
                nodeem = torch.cat([nodes1, torch.Tensor(overlap).unsqueeze(-1).float()], dim=-1)
                # print(nodeem.shape)
                # print(linenodes1.shape)
                xx = torch.cat([nodeem, linenodes1], dim=0)

                l_rtest = len(x['rtest'])
                l_method = len(x['methods'])
                l_ftest = len(x['ftest'])


                Nodes.append(xx.tolist())
                rMask.append(resmask.tolist())
                Res.append(self.pad_seq(res, all_len))
                # inputText.append(self.pad_seq(overlap, self.Nl_Len))
                # LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len))

                lis1 = list()
                lis1.append(nladrow)
                lis1.append(nladcol)
                inputNlad.append(lis1)

                All_len = []
                All_len.append(all_len)
                All_len.append(l_method)
                All_len.append(l_ftest)
                All_len.append(l_rtest)
                al_All_len.append(All_len)
        print("max1: %d max2: %d" % (maxl, maxl2))
        print("correct: %d error: %d" % (correct, error))
        print("error1: %d error2: %d" % (error1, error2))

        # assert(0)#assert(0)
        batchs = [Nodes, inputNlad, Res, rMask, al_All_len]
        self.data = batchs
        open("data/" +self.proj + "data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        # open('nl_voc.pkl', 'wb').write(pickle.dumps(self.Nl_Voc))
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range(len(self.data)):
                if i == 2:
                    # torch.FloatTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])).float()
                    # torch.FloatTensor(self.data[i][offset].data)
                    # torch.FloatTensor(self.data[i][offset].data)
                    # ans.append(self.data[i][offset])
                    # ans.append(torch.sparse.FloatTensor(torch.LongTensor(np.array([self.data[i][offset].row, self.data[i][offset].col])), torch.FloatTensor(self.data[i][offset].data).float(), torch.Size([self.Nl_Len,self.Nl_Len])))
                    # open('tmp.pkl', 'wb').write(pickle.dumps(self.data[i][offset]))
                    # assert(0)
                    ans.append(self.data[i][offset].toarray())
                    # print(self.data[i][offset].toarray()[0, 2545])
                    # assert(0)
                else:
                    ans.append(np.array(self.data[i][offset]))
        else:
            for i in range(len(self.data)):
                if i == 4:
                    continue
                ans.append(np.array(self.data[i][offset]))
            negoffset = random.randint(0, len(self.data[0]) - 1)
            while negoffset == offset:
                negoffset = random.randint(0, len(self.data[0]) - 1)
            if self.dataName == "train":
                ans.append(np.array(self.data[2][negoffset]))
                ans.append(np.array(self.data[3][negoffset]))
        return ans

    def __len__(self):
        return len(self.data[0])

    def del_tensor_ele(self, arr, index):
        arr1 = arr[0:index]
        arr2 = arr[index + 1:]
        return torch.cat((arr1, arr2), dim=0)
    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        sc = 0.3
        for a in range(len(data[0])):
            anss = []
            b_count = 0 #区分正负样本
            for b in range(batch_size):

                b_count+=1
                ans = []
                if b == 0:

                    for c in range(len(data)-1):
                            ans.append(data[c][a])
                    anss.append(ans)
                else:

                    if b_count <= (batch_size/2):

                        #Rtest的节点数
                        #生成随机数
                        # print(data[4])
                        # print(data[4][a])
                        start = data[4][a][1]+data[4][a][2]
                        end = data[4][a][1]+data[4][a][2]+data[4][a][3]
                        if data[4][a][3] == 0:#添加rtest的数量
                            n_num = 10#要添加的rtest的数量
                            #1.节点添加，添加10个rtest
                            l_nod = []
                            for c in range(n_num):
                                l_nod.append('RTest')
                            l_node = torch.Tensor(self.Get_Em(l_nod, self.Nl_Voc)).long()
                            l_node1 = self.token_embedding1(l_node)
                            for c in range(len(data)-1):
                                if c==0:

                                    e = torch.Tensor(data[c][a])
                                    first = e[:start,:]
                                    second = torch.cat([first, l_node1], dim=0)
                                    third = e[start:,:]
                                    d = torch.cat([second, third], dim=0)
                                    ans.append(d)
                                elif c == 1:
                                    ad = []
                                    n_testToNode = int((int(torch.Tensor(data[0][a]).shape[0])-end)*0.3)
                                    line_end = int(torch.Tensor(data[0][a]).shape[0])+9
                                    line_start = end+10
                                    adj1 = torch.tensor(data[c][a][0])
                                    adj2 = torch.tensor(data[c][a][1])
                                    new_adj1 = torch.where(adj1 >= start, adj1.add(10), adj1).tolist()
                                    new_adj2 = torch.where(adj2 >= start, adj2.add(10), adj2).tolist()

                                    for tes_num in range(10):
                                        ln_list = []
                                        for line_for in range(n_testToNode):
                                            rd_num1 = random.randint(line_start,line_end)
                                            if rd_num1 not in ln_list:
                                                ln_list.append(rd_num1)

                                        for line_for in range(len(ln_list)):
                                            new_adj1.append(start+tes_num-1)
                                            new_adj2.append(ln_list[line_for])
                                            new_adj2.append(start+tes_num-1)
                                            new_adj1.append(ln_list[line_for])
                                    ad.append(new_adj1)
                                    ad.append(new_adj2)
                                    ans.append(ad)
                                elif c == 2:
                                    e = torch.Tensor(data[c][a])
                                    first = torch.zeros(10)
                                    d = torch.cat([e, first], dim=0)
                                    ans.append(d)

                                else:
                                    e = torch.Tensor(data[c][a])
                                    first = torch.zeros(10)
                                    second = torch.eq(first,1)
                                    d = torch.cat([e,second], dim=0)
                                    ans.append(d)


                        else:
                            rd_list = []#要删除的节点索引列表
                            while True:
                                rd_num = random.randint(start, end-1)

                                if rd_num not in rd_list:
                                    rd_list.append(rd_num)
                                if len(rd_list)>=sc*data[4][a][3] or data[4][a][3]<=1:
                                    break
                            d_list = []#要保留的节点索引列表
                            for dl in range(data[4][a][0]):
                                if dl not in rd_list:
                                    d_list.append(dl)

                            #根据随机数删除节点并添加到ans中
                            for c in range(len(data)-1):
                                if c==0:

                                    # print("a: ",a)
                                    # print(data[c][a])
                                    # print(d_list)
                                    e = torch.Tensor(data[c][a])
                                    d = torch.Tensor(e[d_list,:])
                                    ans.append(d)
                                elif c == 1:
                                    ad1 = []
                                    ad2 = []
                                    ad = []
                                    for e in range(len(data[c][a][0])):
                                        if data[c][a][0][e] not in rd_list and data[c][a][1][e] not in rd_list:
                                            ad1.append(data[c][a][0][e])
                                            ad2.append(data[c][a][1][e])
                                    ad.append(ad1)
                                    ad.append(ad2)
                                    ans.append(ad)
                                else:
                                    e = torch.Tensor(data[c][a])
                                    d = e[d_list]
                                    ans.append(d)

                    else:

                        start = data[4][a][1]
                        end = data[4][a][1] + data[4][a][2]
                        rd_list = []  # 要删除的节点索引列表
                        while True:
                            rd_num = random.randint(start, end - 1)
                            if rd_num not in rd_list:
                                rd_list.append(rd_num)
                            if len(rd_list) >= sc * data[4][a][2] or data[4][a][2]<=1:
                                break
                        d_list = []  # 要保留的节点索引列表
                        for dl in range(data[4][a][0]):
                            if dl not in rd_list:
                                d_list.append(dl)
                        # 根据随机数删除节点并添加到ans中
                        for c in range(len(data) - 1):
                            if c == 0:
                                e = torch.Tensor(data[c][a])
                                d = e[d_list,:]
                                ans.append(d)
                            elif c == 1:
                                ad1 = []
                                ad2 = []
                                ad = []
                                for e in range(len(data[c][a][0])):
                                    if data[c][a][0][e] not in rd_list and data[c][a][1][e] not in rd_list:
                                        ad1.append(data[c][a][0][e])
                                        ad2.append(data[c][a][1][e])
                                ad.append(ad1)
                                ad.append(ad2)
                                ans.append(ad)
                            else:
                                e = torch.Tensor(data[c][a])
                                d = e[d_list]
                                ans.append(d)
                    anss.append(ans)

            ans1 = []
            for b in range(len(anss[0])):

                tmp_d = None
                l = []
                l1 = []
                l2 = []

                for bb in range(len(anss)):

                    if b != 1:


                        if bb == 0:
                            tmp_d = torch.Tensor(anss[bb][b])
                        else:
                            tmp_d = torch.cat([torch.Tensor(tmp_d), torch.Tensor(anss[bb][b])], dim=0)

                    else:

                        count_num = 0
                        for d in range(len(anss[bb][b])):
                            if d % 2 == 0:
                                # print("anss[b][c][d] : ",anss[bb][b][d])
                                for e in anss[bb][b][d]:
                                    l1.append(e + count_num)
                            elif d % 2 == 1:
                                for e in anss[bb][b][d]:
                                    l2.append(e + count_num)
                            else:
                                print(1111111111111)

                        count_num += data[4][a][0]
                if tmp_d != None:
                    ans1.append(tmp_d)
                else:
                    l.append(l1)
                    l.append(l2)
                    ans1.append(l)
            yield ans1


class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=1, concat=False).to('cuda:0')
        self.conv2 = GATConv(h_feats, out_feats, heads=1, concat=False).to('cuda:1')

    def forward(self, x, inputad):
        x, edge_index = x, inputad
        print(x.shape)
        x = self.conv1(x.to('cuda:0'), edge_index.to('cuda:0'))
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x.to('cuda:1'), edge_index.to('cuda:1'))
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        
        self.feed_forward_hidden = 4 * self.embedding_size
        self.GATBlocks = GAT(self.embedding_size, self.embedding_size, self.embedding_size)
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 1)
        self.token_embedding1 = nn.Embedding(args.Nl_Vocsize, self.embedding_size)
        self.text_embedding = nn.Embedding(20, self.embedding_size)
        self.resLinear2 = nn.Linear(self.embedding_size, 1).to('cuda:2')

    def forward(self, input_node, inputad, res, resmask):
        x = self.GATBlocks.forward(input_node, inputad)
        # print(x.shape)
        a = int(resmask.sum())
        # print("a: ",a)
        # print(resmask.shape)
        mask = resmask.repeat(self.embedding_size,1)
        mask = mask.permute(1,0)
        # print(mask.shape)
        mask = torch.eq(mask,1).to('cuda:2')
        # print(x.shape)
        x = x.to('cuda:2')
        select = x.masked_select(mask)
        # print(select.shape)
        s = select.view(a, self.embedding_size)
        li = self.resLinear2(x)
        li = li.squeeze(-1)
        resmask = resmask.to('cuda:2')
        li = li.masked_fill(resmask == 0, -1e9)
        resSoftmax = F.softmax(li, dim=-1)
        res = res.to('cuda:2')
        loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * res
        loss = loss.sum(dim=-1)
        return loss, resSoftmax, x, resmask,s





# NlLen_map = {"Time": 3900, "Math": 4500, "Lang": 280, "Chart": 2350, "Mockito": 1780, "unknown": 2200}
# CodeLen_map = {"Time": 1300, "Math": 2700, "Lang": 300, "Chart": 5250, "Mockito": 1176, "unknown": 2800}

os.environ['PYTHONHASHSEED'] = str(args.seed)


def save_model(model, dirs="checkpointcodeSearch"):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    torch.save(model.state_dict(), dirs + '/best_model.ckpt')


def load_model(model, dirs="checkpointcodeSearch"):
    assert os.path.exists(dirs + '/best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + '/best_model.ckpt'))


use_cuda = torch.cuda.is_available()


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        try:
            tensor = torch.from_numpy(data)
        except:
            print(tensor)
    elif isinstance(data, list):
        # tensor = torch.Tensor(data).long()
        for i in range(len(data)):
            data[i] = gVar(data[i])
        tensor = data
    # else:
    #     print(type(tensor))
    # print(tensor)
    # assert isinstance(tensor, torch.Tensor)
#     if use_cuda:

#         tensor = tensor.cuda()

    return tensor


def train(t=5, p='Math'):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed + t)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev_set = SumDataset(args, "test", p, testid=t)
    val_set = SumDataset(args, "val", p, testid=t)
    data = pickle.load(open("data/" + p + '.pkl', 'rb'))
    dev_data = pickle.load(open("data/" + p + '.pkl', 'rb'))
    train_set = SumDataset(args, "train", testid=t, proj=p, lst=dev_set.ids + val_set.ids)

    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)

    print(dev_set.ids)
    model = NlEncoder(args)
#     if use_cuda:
#         print('using GPU')
#         model = nn.DataParallel(model.cuda(),device_ids=[0,1,2,3])

    maxl = 1e9
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=args.lr), args.embedding_size, 4000)
    maxAcc = 0
    minloss = 1e9
    rdic = {}
    brest = []
    bans = []
    batchn = []
    each_epoch_pred = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    for epoch in range(15):
        # lossesss = 0
        index = 0

        for dBatch in tqdm(train_set.Get_Train(args.batch_size)):

            if index == 0:
                accs = []
                loss = []
                model = model.eval()

                score2 = []

                for k, devBatch in tqdm(enumerate(val_set.Get_Train(len(val_set)))):
                    for i in range(len(devBatch)):
                        if i == 1:
                            devBatch[i] = torch.Tensor(devBatch[i]).long()
#                             if use_cuda:
#                                 devBatch[i] = devBatch[i].cuda()

                        else:
                            devBatch[i] = gVar(devBatch[i])
                    with torch.no_grad():
                        print("len(devBatch):   :",len(devBatch))

                        l, pre, _, resmask,sele = model(devBatch[0], devBatch[1], devBatch[2], devBatch[3])

                        l_score = []
                        l_scores = []
                        pre = -pre
                        pre = pre.masked_fill(resmask == 0, 1e9)
                        pre = pre.tolist()
                        # print(pre)
                        flag = 0
                        for iii in pre:
                            if int(iii) != int(1000000000.0):
                                flag = 1
                            elif flag == 1 and int(iii) == int(1000000000.0):
                                flag = 2

                            if flag == 1:
                                l_score.append(iii)
                            elif flag == 2:
                                l_scores.append(np.argsort(np.asarray(l_score)))
                                l_score = []
                                flag = 3
                        #                         print("l_scores: ", l_scores)  # 所有的方法的排序列表Ranklist
                        #
                        #
                        #
                        # s = -pre#-pre[:, :, 1]
                        # s = s.masked_fill(resmask == 0, 1e9)
                        # print(s)
                        # pred = s.argsort(dim=-1)
                        # print(pred)
                        # pred = pred.data.cpu().numpy()
                        # alst = []
                        # print(pred)

                        for k in range(len(l_scores)):
                            datat = data[val_set.ids[k]]
                            # print(datat['ans'])#有故障的方法的编号
                            maxn = 1e9
                            lst = l_scores[k].tolist()  # score = np.sum(loss) / numt
                            # bans = lst
                            for x in datat['ans']:
                                i = lst.index(x)
                                maxn = min(maxn, i)
                            score2.append(maxn)

                each_epoch_pred[epoch] = lst
                score = score2[0]
                print('curr accuracy is ' + str(score) + "," + str(
                    score2))  # 有故障的方法（datat['ans']）在Ranklist（l_scores）中的排名（从0开始）
                if score2[0] == 0:
                    batchn.append(epoch)

                if maxl >= score:
                    brest = score2
                    bans = lst
                    maxl = score
                    print("find better score " + str(score) + "," + str(score2))
                    # save_model(model)
                    # torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
                model = model.train()

            for i in range(len(dBatch)):
                # print(111111111)
                if i == 1:
                    dBatch[i] = torch.Tensor(dBatch[i]).long()
#                     if use_cuda:
#                         dBatch[i] = dBatch[i].cuda()
                else:
                    dBatch[i] = gVar(dBatch[i])
                # dBatch[i] = gVar(dBatch[i])
            loss, _, _, resmask,select = model(dBatch[0], dBatch[1], dBatch[2], dBatch[3])
            n_z = int(select.shape[0]/2)
            z1 = select[0:n_z,:].to('cuda:2')
            z2 = select[n_z:,:].to('cuda:2')
            # print(loss.mean().item())
            optimizer.zero_grad()
            loss = loss.mean()
#             loss1 = GraphLoss(args.embedding_size,args.embedding_size*2).cuda()
            loss1 = GraphLoss(args.embedding_size,args.embedding_size*2).to('cuda:2')
            loss2 = loss1.forward(z1,z2,batch_size=args.batch_size)
            loss3 = loss2+loss
            print(loss3)
            loss3.backward()
            optimizer.step_and_update_lr()
            index += 1
    return brest, bans, batchn, each_epoch_pred


if __name__ == "__main__":
    args.lr = 1e-2
    args.seed = 0
    args.batch_size = 4

    np.set_printoptions(threshold=sys.maxsize)
    res = {}

    dict1 = {
#         "Lang": 62,
#         "Math": 104,
        "Closure": 119,
        # "Chart": 25,
        # "Mockito": 36,
        # "Time": 25,
    }
    for k in dict1:
        p = k
        num = 14
        for i in range(dict1[k]):
            res[num] = train(num, p)
            open("data/" + p + '/%sres%d_%d_%s_%s.pkl' % (p, num, args.seed, args.lr, args.batch_size), 'wb').write(
                pickle.dumps(res))
            num += 1



