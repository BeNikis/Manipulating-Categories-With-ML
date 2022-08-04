#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:07:01 2022

@author: benikis
"""
import CTAI.framework as CT
import lark
from hypothesis.extra.lark import from_lark
import torch
from torch.nn import RNN
from random import choice

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

def gen_queries(n,eqs,ms,simple=True):
    text = ""
    qs = choice (["C?","T?"])
    if simple: #generate very simple queries
        for i in range(n):
            q = choice (qs)
            text += q+" "
            if q=="C?":
                
                text+=choice(eqs)+"\n"
            elif q=="T?":
                chosen_eq=choice(eqs)
                typ = chosen_eq[chosen_eq.index('='):].strip()
                typ = ms[ms.index(typ)].tgt.name
                text += choice([(lambda eq:eq[:eq.index("=")])(chosen_eq)])+typ+"\n"
    else: #combine equations
        #Composition queries only for now
        for i in range(n):
            split = lambda eq:list(map (lambda comp:comp.split(' '),eq.split('=')))
            split_eqs = list(map(split,eqs))
            lhss = list(map(lambda eq:eq[0],split_eqs))
            rhss = list(map(lambda eq:eq[1],split_eqs))
            
            eq = choice(split_eqs)
            
            for m_i,m in enumerate(eq[0]):
                for s_i,swap_candidate in enumerate(rhss):
                    if m==swap_candidate and choice([True,False]):
                        eq[0][m_i]=" ".join(lhss[s_i])
                        if choice([True,False]):
                            break
            text+="C? "+" ".join(eq[0])+"="+" ".join(eq[1])+"\n"
            
    return text
def gen_cat_text(C:CT.Category,gen_qs=False):

    ms = C.all_morphisms()
    
    text = "MORS\n"
    for m in ms:
        text += m.name+" : "+m.src.name+" "+m.tgt.name+"\n"
    
    text += "EQS\n"
    eqs = []      
    for n in range(len(ms)):
         lhs=[choice(ms)]
         for i in range(1,choice(range(4))+3):
             while C.one_step(lhs[-1].tgt)==[]:
                 lhs[-1]=choice(ms)
             lhs.append(choice(C.one_step(lhs[-1].tgt)))
         
        
         eq = " ".join(map(lambda m:m.name,lhs))
         eq += " = "+choice(C.ms[lhs[0].src][lhs[-1].tgt]).name
         eqs.append(eq)
    text += "\n".join(eqs)
    if gen_qs:
        text += "\nQUERIES\n"
    
        text += gen_queries(10,eqs,ms,False)    
            
            
    return text #make more difficult later 
        
grammar = open("CT_1.lark",'r')

#instantiate with the grammar and embedding in/out sizes (for tokens),embed tree (parsed string) with .embed(prsd)
class GrammarPreembedding:
    rules = []
    r_count =0
    
    terminals = []
    t_count = 0
    
    flattened_tree=[] #holds the flattened form of last embedded tree
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
    
        
    
    def embed(self,tree:lark.Tree,first_call=True):
        flat =[]
        if first_call:
            self.flattened_tree=[]
        #skip 'start' rule if parsing complete tree
        if not tree.data=='start':
            
            flat.append(self.embed_single_symbol(1,self.rules.index(tree.data),tree.data))
            self.flattened_tree.append(lark.Token('RULE',tree.data))
        
        for c in tree.children:
            if type(c)==lark.Tree:
                flat+=self.embed(c,False)
            else:
                flat.append(self.embed_single_symbol(0,self.terminals.index(c.type),c))
                self.flattened_tree.append(c)
        
        return torch.vstack(flat) if first_call else flat
    
    
    def embed_single_symbol(self,rule_or_term,type_index,s):
        #print(rule_or_term,type_index,s)
        from string import ascii_letters
        symbols=ascii_letters+'_\n'
        index = { l:i/(len(symbols)+1) for i,l in enumerate(symbols)}
        
        enc_s=torch.zeros(len(s),1)
        for i,c in enumerate(s):
            enc_s[i,0]=index[c]
        embd_s=self.rnn(enc_s)[1]
        
        #shape is token type+one-hot rule type+one-hot terminal type
        out_token = torch.zeros(self.embd_size)
        out_token[0]=rule_or_term
        
        if rule_or_term==1:
            out_token[1+type_index]=1
        else:
            out_token[1+self.r_count+type_index]=1
        
        

        offset = self.type_in_size
        for i,h_v in enumerate(embd_s[0,:]):
            out_token[offset+i]=h_v
                
            
            
            
        return out_token
        
     
        
CT_p = lark.Lark(grammar)
CT_pre = GrammarPreembedding(CT_p)
        
            
        


if __name__=="__main__":

    
    
    #print(parser.parse("MORS f g h EQS f g=h QUERY C?f g = h").pretty())
    gened = gen_cat_text(CT.gen_abstract_category(4, 3),True)
    print(gened)

    p=CT_p.parse(gened)
    print(p,"\n-------\n")
    
    print('\n')
    print(CT_pre.rules,'\n')
    print(CT_pre.terminals)
    

    print(CT_pre.flattened_tree)
    print(CT_pre.embed(p).shape)
    

    #gen_from_lark(parser,100,"typing")

