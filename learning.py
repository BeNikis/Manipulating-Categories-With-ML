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
query_batch_size = 10

p = prepro.CT_p
emb= prepro.CT_pre
#outputs a single value based on hidden state of controller.Example use:treat the output as the probability that that the controller has finished it's job
class Output(nn.Module):
    def __init__(self,controller:nn.Module):
        super().__init__()
        self.c = controller
        if type(controller)==nn.RNN:
            self.in_size=controller.hidden_size*controller.num_layers
        elif type(controller)==RNNWithFinishAndMem:
            
            #TODO correctly calculate the shape of input
            self.in_size=self.c.hidden_size*self.c.num_layers+self.c.tape_out_size
            #print("o ",self.in_size)
            
        
        
        self.network=nn.Sequential(*[nn.Linear(self.in_size, 32),nn.ReLU(),nn.Linear(32, 16),nn.ReLU(),nn.Linear(16, 1),nn.Sigmoid()])
        
    def forward(self):
        #print(self.in_size,self.c.mems[0].tapes.shape)
        tapes = torch.zeros(self.c.tape_out_size)
        t_tapes = torch.cat(list(map(lambda mem:mem.tapes.reshape(-1),self.c.mems)))
        tapes[:t_tapes.shape[0]]=t_tapes 
        inp=torch.cat([self.c.c_hid.reshape(-1),tapes])
        #print('os',inp.shape)
        
        return self.network(inp)
    

class RNNWithFinishAndMem(nn.Module):
    def __init__(self,embedder:prepro.GrammarPreembedding,num_mems:int,hidden_size,num_layers:int):
        super().__init__()

        self.emb = embedder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mems = num_mems
        self.tape_out_size=batch_size*self.emb.embd_size*self.num_mems*2
        self.mems=[Mem.empty(batch_size,self.emb.embd_size,2) for i in range(num_mems)]
        self.mem_policies=[0 for i in range(num_mems)]
        self.mem_new_vecs=[0 for i in range(num_mems)]
        

        self.controller = nn.RNN(self.emb.embd_size,hidden_size,num_layers)
        self.c_hid = 0#torch.zeros(num_layers,hidden_size)
        self.c_out = 0
        
        self.policy_networks=nn.ModuleList([nn.Sequential(nn.Linear(hidden_size*num_layers,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,self.mems[0].get_num_actions())) for i in range (num_mems)])
        self.new_vec_networks=nn.ModuleList([nn.Sequential(nn.Linear(hidden_size*num_layers,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,self.emb.embd_size)) for i in range (num_mems)])
        
        self.finished = Output(self)
        
        return
    
    def forward(self,embedded_tree):
        
        self.c_out,self.c_hid = self.controller(embedded_tree)
        print(self.c_out.shape,self.c_hid.shape)

        for i in range(len(self.mems)):
           #print ('p')
            policy = self.policy_networks[i](self.c_hid)
            #print ('v')
            new_vecs=self.new_vec_networks[i](self.c_hid)
            print(policy.shape,new_vecs.shape)
            self.mems[i].update(policy,new_vecs)
            
        return self.c_out,self.c_hid,list(map(lambda mem:mem.tapes,self.mems)),self.finished()
        
        
def get_query_batch(gen:prepro.CategoryTextGenerator,emb:prepro.GrammarPreembedding,n,simple=True):
    queries,_,_ =  gen.gen_queries(n,simple)
    embedding = list(map(emb.embed,map(lambda q:p.parse(q,start='c_eq'),queries)))
    targets = map(lambda emb_q:emb_q[-1],embedding)
    embedding = map(lambda emb_q:emb_q[:-1],embedding)
    
    #print(queries[-1],embedding[-1])
    #print(list(gen.split_qs))
    return embedding,targets
    
    
         
                     
        
        
       
class DiffCWithOut(nn.Module):
    def __init__(self,grammar:prepro.GrammarPreembedding):
        self.diff_c=DIFFC(grammar.embd_size, 64)
        self.out=Output()
        
if __name__=="__main__":
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    #loss = 
    
    stackrnn = RNNWithFinishAndMem(emb, 3, 64, 1)
    
    #max_seq_size=0
    seqs=[]
    targets = torch.zeros((batch_size,query_batch_size,emb.embd_size))
    
    #generate batch/targets
    for i in range(batch_size):
        gen = prepro.CategoryTextGenerator(CT.gen_abstract_category(4, 3))
        gened = gen.get_text(False)
        embedded=(emb.embed(p.parse(gened,start='start')))

        (s_queries,targets) = get_query_batch(gen,emb,query_batch_size)
        c_out,c_hid,tapes,finished=stackrnn(embedded)
        definition_hidden = stackrnn.c_hid
        #print(list(s_queries),list(targets))
        n_targ = query_batch_size
        #max_seq_size=max(max_seq_size,seqs[-1].shape[0])
    
    
    #print(batch.shape)

    
    """
    rnn = nn.RNN(prepro.CT_pre.embd_size,32)
    o = Output(rnn) 
    print(o(rnn(prepro.CT_pre.embed(prepro.CT_p.parse(prepro.gen_cat_text(CT.gen_abstract_category(3, 5)))))[1]))
    """


     