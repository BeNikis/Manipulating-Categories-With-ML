#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:50:06 2022

@author: benikis
"""
import CTAI.framework as CT
import preprocessing as prepro

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from stacknn.superpos import NoOpStack as Mem

import matplotlib.pyplot as plt
from functools import reduce 
from dnc import SAM as DIFFC

batch_size=16
#outputs a single value based on hidden state of controller.Example use:treat the output as the probability that that the controller has finished it's job
class Output(nn.Module):
    def __init__(self,controller:nn.Module):
        super().__init__()
        self.c = controller
        if type(controller)==nn.RNN:
            self.in_size=controller.hidden_size*controller.num_layers
        elif type(controller)==RNNWithFinishAndMem:
            
            #TODO correctly calculate the shape of input
            self.in_size=12544#self.c.hidden_size*self.c.num_layers+reduce(lambda x,y:x*y,self.c.mems[0].tapes.shape)*self.c.num_mems
            #print("o ",self.in_size)
            
        
        
        self.network=nn.Sequential(*[nn.Linear(self.in_size, 32),nn.ReLU(),nn.Linear(32, 16),nn.ReLU(),nn.Linear(16, 1),nn.Sigmoid()])
        
    def forward(self):
        #print(self.in_size,self.c.mems[0].tapes.shape)
        inp=torch.cat([self.c.c_hid.reshape(-1)]+list(map(lambda mem:mem.tapes.reshape(-1),self.c.mems)))
        #print('os',inp.shape)
        
        return self.network(inp)
    

class RNNWithFinishAndMem(nn.Module):
    def __init__(self,embedder:prepro.GrammarPreembedding,num_mems:int,hidden_size,num_layers:int):
        super().__init__()

        self.emb = embedder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mems = num_mems
        
        self.mems=[Mem.empty(batch_size,self.emb.embd_size,3) for i in range(num_mems)]
        self.mem_policies=[0 for i in range(num_mems)]
        self.mem_new_vecs=[0 for i in range(num_mems)]
        
        self.c_hid = 0
        self.c_out = 0
        self.controller = nn.RNN(self.emb.embd_size,hidden_size,num_layers)
        self.policy_networks=nn.ModuleList([nn.Sequential(nn.Linear(hidden_size*num_layers,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,self.mems[0].get_num_actions())) for i in range (num_mems)])
        self.new_vec_networks=nn.ModuleList([nn.Sequential(nn.Linear(hidden_size*num_layers,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,self.emb.embd_size)) for i in range (num_mems)])
        
        self.finished = Output(self)
        
        return
    
    def forward(self,embedded_tree):
        
        self.c_out,self.c_hid = self.controller(embedded_tree)
        #print(self.c_out.shape,self.c_hid.shape)
        self.c_hid = self.c_hid
        for i in range(len(self.mems)):
            policy = self.policy_networks[i](self.c_hid)
            new_vecs=self.new_vec_networks[i](self.c_hid)
            
            self.mems[i].update(policy.reshape(batch_size,self.mems[i].get_num_actions()),new_vecs)
            
        return self.c_out,self.c_hid,list(map(lambda mem:mem.tapes,self.mems)),self.finished()
        
        
        
        
        
        
       
class DiffCWithOut(nn.Module):
    def __init__(self,grammar:prepro.GrammarPreembedding):
        self.diff_c=DIFFC(grammar.embd_size, 64)
        self.out=Output()
        
if __name__=="__main__":
    p = prepro.CT_p
    emb= prepro.CT_pre
    
    stackrnn = RNNWithFinishAndMem(emb, 3, 64, 1)
    
    max_seq_size=0
    seqs=[]
    for i in range(batch_size):
        gened = prepro.CategoryTextGenerator(CT.gen_abstract_category(4, 3)).get_text(True)
        seqs.append(emb.embed(p.parse(gened)))
        max_seq_size=max(max_seq_size,seqs[-1].shape[0])
    
    batch=nn.utils.rnn.pad_sequence(seqs)
    print(batch.shape)
    c_out,c_hid,tapes,finished=stackrnn(batch)
    
    """
    rnn = nn.RNN(prepro.CT_pre.embd_size,32)
    o = Output(rnn) 
    print(o(rnn(prepro.CT_pre.embed(prepro.CT_p.parse(prepro.gen_cat_text(CT.gen_abstract_category(3, 5)))))[1]))
    """


     