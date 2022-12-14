#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:07:01 2022

@author: benikis
"""
from copy import copy
import CTAI.framework as CT
import lark
from hypothesis.extra.lark import from_lark
import torch
from torch.nn import RNN
import torch
from random import choice
from scipy.spatial import KDTree

from string import ascii_letters
symbols=ascii_letters+'_\n'

class CategoryTextGenerator:
    c : CT.Category
    definition : str #text of definion (MORS+EQS)
    
    ms: list #morphisms
    eqs: list 
    qs: list 
    
    
    def __init__(self,C:CT.Category):
        self.c = C
        self.ms = C.all_morphisms()
        self.eqs = []
        self.qs = []
        self.split_eq = lambda eq:list((map (lambda comp:list(filter(lambda s:not s=='',comp.split(' '))),eq.split('='))))
        self.split_qs = []

        
        
        text = "MORS\n"
        for m in self.ms:
            text += m.name+" : "+m.src.name+" "+m.tgt.name+"\n"
        
        text += "EQS\n"
 
        for n in range(len(self.ms)):
             lhs=[choice(self.ms)]
             for i in range(1,choice(range(4))+3):
                 while C.mors_from(lhs[-1].tgt)==[]:
                     lhs[-1]=choice(self.ms)
                 lhs.append(choice(C.mors_from(lhs[-1].tgt)))
             
            
             eq = " ".join(map(lambda m:m.name,lhs))
             eq += " = "+choice(C.ms[lhs[0].src][lhs[-1].tgt]).name
             self.eqs.append(eq)
        text += "\n".join(self.eqs)
        #print(self.eqs)
        self.definition = copy(text)
        return
 
    def gen_queries(self,n,simple=True):
        text = "\nQUERIES\n"
        qs = choice (["C?"])#],"T?"])
    
        self.qs=[]
        
        if simple: #generate very simple queries
            for i in range(n):
                q = choice (qs)
                text += "C? "
                #if q=="C?":
                chosen_eq = choice(self.eqs)
                text+=chosen_eq+"\n"
                self.qs.append(chosen_eq)
                self.split_qs.append(self.split_eq(chosen_eq))
                # elif q=="T?":
                #     chosen_eq=choice(self.eqs)
                #     typ = chosen_eq[chosen_eq.index('='):].strip()
                #     typ = self.ms[self.ms.index(typ)].tgt.name
                #     text += choice([(lambda eq:eq[:eq.index("=")])(chosen_eq)])+typ+"\n"
        else: #combine equations
            #Composition queries only for now
            self.split_qs=[]
            for i in range(n):
                
                split_eqs=list(map(self.split_eq,self.eqs))
                lhss = list(map(lambda eq:eq[0],split_eqs))
                rhss = list(map(lambda eq:eq[1],split_eqs))
                
                eq = choice(split_eqs)
                
                for m_i,m in enumerate(eq[0]):
                    for s_i,swap_candidate in enumerate(rhss):
                        if m==swap_candidate and choice([True,False]):
                            eq[0][m_i]=" ".join(lhss[s_i])
                            if choice([True,False]):
                                break
                            
                self.qs.append(" ".join(eq[0])+"="+" ".join(eq[1]))
                self.split_qs.append(eq)
                text+=self.qs[-1]+'\n'
                #print(text)
        
        #print(self.split_qs)
        return self.qs,self.split_qs,text
 
    def get_text(self,queries=False,simple_queries=True):
        return self.definition+(self.gen_queries(10,simple_queries)[2] if queries else "")
                                               #fix for typing
     
        
       


def gen_from_lark(lark,n,start="start"):
    gen = from_lark(lark,start=start)
    gend=[]
    for i in range(n):
        accd = False
        
        while not accd:
            g = gen.example()
            if g not in gend:
                gend.append(g)
                print(g,"\n---------\n")
                accd = True



            
            
    return text #make more difficult later 
        
grammar = open("CT_1.lark",'r')

#instantiate with the grammar and embedding in/out sizes (for tokens),embed tree (parsed string) with .embed(prsd)
class GrammarPreembedding:
    rules = []
    r_count =0
    
    terminals = []
    t_count = 0
    
    flattened_tree=[] #holds the flattened form of last embedded tree
    embedded_flattened_tree=[]
    kd_tree = None
    def __init__(self,parser:lark.Lark,embed_out_size:int=64,n_layers=1):
        for t in parser.terminals:
            if (not t.name[0]=='_') and t.name not in self.terminals : #skip terminals that we skip in the tree or are defined as helpers by lark.
                self.terminals.append(t.name)
                self.t_count+=1
        for r in parser.rules:
            if (not r.origin.name[0]=='_') and r.origin.name not in self.rules: #same but for rules
                self.rules.append(r.origin.name)
                self.r_count+=1
        self.type_in_size=1+self.r_count+self.t_count #the additional dimension is for type of token - 1 for rule or 0 for terminal
        self.symbol_in_size=embed_out_size
        self.embd_size=embed_out_size+self.type_in_size
        self.rnn=RNN(1,self.symbol_in_size,1)
        
        self.index = { l:i/(len(symbols)+1) for i,l in enumerate(symbols)}
    
        
    
    def embed(self,tree:lark.Tree,first_call=True,emb_def=False):
        flat =[]
        embedding_definition = False
        
        #skip 'start' rule if parsing complete tree
        if not tree.data=='start':
            
            flat.append(self.embed_single_symbol(1,self.rules.index(tree.data),tree.data))
            if emb_def:
                self.flattened_tree.append(lark.Token('RULE',tree.data))
                self.embedded_flattened_tree.append(flat[-1])
            
        else: #embedding the definition of a category so generate a lookup kd-tree
            embedding_definition = True
            emb_def=True
            if first_call:
                self.flattened_tree=[]
        for c in tree.children:
            if type(c)==lark.Tree:
                flat+=self.embed(c,False,embedding_definition)
                #flat.append(self.embed_single_symbol(-1,self.rules.index(c.data),c.data))
                #self.embedded_flattened_tree.append(self.embed_single_symbol(-1,self.rules.index(c.data),c.data))
            else:
                flat.append(self.embed_single_symbol(0,self.terminals.index(c.type),c))
                
                if emb_def:
                    self.flattened_tree.append(c)
                    self.embedded_flattened_tree.append(flat[-1])
        
        #if embedding_definition and first_call:
            #print(len(self.flattened_tree),len(self.embedded_flattened_tree))
            #self.kd_tree = KDTree(list(map(lambda tok:tok.detach().numpy(),flat)))
            #print(self.kd_tree.size,self.kd_tree.m,self.kd_tree.n)
        return torch.vstack(flat) if first_call else flat
    
    
    def embed_single_symbol(self,rule_or_term,type_index,s):
        #print(rule_or_term,type_index,s)
        
        
        
        enc_s=torch.zeros(len(s),1,requires_grad=False)
        for i,c in enumerate(s):
            enc_s[i,0]=self.index[c]
        embd_s=self.rnn(enc_s)[0]
        # embd_s=torch.abs(embd_s)
        # embd_s=embd_s/torch.max(embd_s) #normalize
        #shape is token type+one-hot rule type+one-hot terminal type
        out_token = torch.ones(self.embd_size,requires_grad=False)
        out_token[0]=rule_or_term
        
        if rule_or_term==1:
            out_token[1+type_index]=1
        else:
            out_token[1+self.r_count+type_index]=1
        
        

        offset = self.type_in_size
        for i,h_v in enumerate(embd_s[0,:]):
            out_token[offset+i]=h_v
        
            
            
            
        return torch.Tensor(out_token)
        
    def lookup_token(self,embedded_token):
        min_length = 99999999.0
        ret_tok = 0
        mse = torch.nn.MSELoss()
        for i,tok in enumerate(self.flattened_tree):
            dist = mse(self.embedded_flattened_tree[i],embedded_token)
            #print(dist.item(),ret_tok)
            if dist<min_length:
                ret_tok=tok
                min_length=dist
        return ret_tok
                
         # nn = self.kd_tree.query(embedded_token.detach().numpy())
         # print(nn)
         # return self.flattened_tree[nn[1]]
         
        
        
CT_p = lark.Lark(grammar,start=['start','queries','c_eq'])
CT_pre = GrammarPreembedding(CT_p)
        
            
        


if __name__=="__main__":
    gen = CategoryTextGenerator(CT.gen_abstract_category(4, 3))
    gened=gen.get_text(False)
    
    embedded=(CT_pre.embed(CT_p.parse(gened,start='start')))
    
    token=CT_pre.lookup_token(choice(CT_pre.embedded_flattened_tree))
    print(token)
    

    #gen_from_lark(parser,100,"typing")

