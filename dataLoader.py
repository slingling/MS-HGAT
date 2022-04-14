# Part of this file is derived from 
# https://github.com/albertyang33/FOREST
"""
Created on Mon Jan 18 22:28:02 2021

@author: Ling Sun
"""
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle


class Options(object):
    
    def __init__(self, data_name = 'douban'):
        self.data = 'data/'+data_name+'/cascades.txt'
        self.u2idx_dict = 'data/'+data_name+'/u2idx.pickle'
        self.idx2u_dict = 'data/'+data_name+'/idx2u.pickle'
        self.save_path = ''
        self.net_data = 'data/'+data_name+'/edges.txt'
        self.embed_dim = 64

def Split_data(data_name, train_rate =0.8, valid_rate = 0.1, random_seed = 300, load_dict=True, with_EOS=True):
        options = Options(data_name)
        u2idx = {}
        idx2u = []
        if not load_dict:
            user_size, u2idx, idx2u = buildIndex(options.data)
            with open(options.u2idx_dict, 'wb') as handle:
                pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(options.idx2u_dict, 'wb') as handle:
                pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(options.u2idx_dict, 'rb') as handle:
                u2idx = pickle.load(handle)
            with open(options.idx2u_dict, 'rb') as handle:
                idx2u = pickle.load(handle)
            user_size = len(u2idx)
            
        t_cascades = []
        timestamps = []
        for line in open(options.data):
            if len(line.strip()) == 0:
                continue
            timestamplist = []
            userlist = []
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    # Twitter,Douban
                    if len(chunk.split())==2:
                        user, timestamp = chunk.split()
                    # Android,Christianity
                    elif len(chunk.split())==3:
                        root, user, timestamp = chunk.split()                                           
                        if root in u2idx:          
                            userlist.append(u2idx[root])                        
                            timestamplist.append(float(timestamp))
                except:
                    print(chunk)
                if user in u2idx:
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))

            if len(userlist) > 1 and len(userlist)<=500:
                if with_EOS:
                    userlist.append(Constants.EOS)
                    timestamplist.append(Constants.EOS)
                t_cascades.append(userlist)
                timestamps.append(timestamplist)
                
        
        '''ordered by timestamps'''        
        order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x:x[1])]
        timestamps = sorted(timestamps)
        t_cascades[:] = [t_cascades[i] for i in order]
        cas_idx = [i for i in range(len(t_cascades))]
        
        '''data split'''
        train_idx_ = int(train_rate*len(t_cascades))
        train = t_cascades[0:train_idx_]
        train_t = timestamps[0:train_idx_]
        train_idx = cas_idx[0:train_idx_]
        train = [train, train_t, train_idx]
        
        valid_idx_ = int((train_rate+valid_rate)*len(t_cascades))
        valid = t_cascades[train_idx_:valid_idx_]
        valid_t = timestamps[train_idx_:valid_idx_]
        valid_idx = cas_idx[train_idx_:valid_idx_]
        valid = [valid, valid_t, valid_idx]
        
        test = t_cascades[valid_idx_:]
        test_t = timestamps[valid_idx_:]
        test_idx = cas_idx[valid_idx_:]
        test = [test, test_t, test_idx]
            
        random.seed(random_seed)
        random.shuffle(train)
        random.seed(random_seed)
        random.shuffle(train_t)
        random.seed(random_seed)
        random.shuffle(train_idx)
        
        total_len =  sum(len(i)-1 for i in t_cascades)
        train_size = len(train_t)
        valid_size = len(valid_t)
        test_size = len(test_t)
        print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
        print("total size:%d " %(len(t_cascades)))
        print("average length:%f" % (total_len/len(t_cascades)))
        print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
        print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))    
        print("user size:%d"%(user_size-2))           
        
        return user_size, t_cascades, timestamps, train, valid, test
    
def buildIndex(data):
        user_set = set()
        u2idx = {}
        idx2u = []

        lineid=0
        for line in open(data):
            lineid+=1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split(',')
            for chunk in chunks:
                try:
                    if len(chunk.split())==2:
                        user, timestamp = chunk.split()
                    elif len(chunk.split())==3:
                        root, user, timestamp = chunk.split()                                           
                        user_set.add(root)
                except:
                    print(line)
                    print(chunk)
                    print(lineid)
                user_set.add(user)
        pos = 0
        u2idx['<blank>'] = pos
        idx2u.append('<blank>')
        pos += 1
        u2idx['</s>'] = pos
        idx2u.append('</s>')
        pos += 1

        for user in user_set:
            u2idx[user] = pos
            idx2u.append(user)
            pos += 1
        user_size = len(user_set) + 2
        print("user_size : %d" % (user_size))
        return user_size, u2idx, idx2u
        
class DataLoader(object):
    ''' For data iteration '''

    def __init__(
        self, cas, batch_size=64, load_dict=True, cuda=True,  test=False, with_EOS=True): 
        self._batch_size = batch_size
        self.cas = cas[0]
        self.time = cas[1]
        self.idx = cas[2]
        self.test = test
        self.with_EOS = with_EOS          
        self.cuda = cuda
        
        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

     

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = 200
            
            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst)) if len(inst)<max_len else inst[:max_len]  
               for inst in insts])
                
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            seq_idx = Variable(
                torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)
            
            return seq_data, seq_data_timestamp, seq_idx
        else:

            self._iter_count = 0
            raise StopIteration()
