# Import libraries
import os
import pickle
import ujson
from spacy.lang.en import English
tokenizer = English().tokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback

from config import base_config
use_gpu = base_config.get("use_gpu",False)
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu == True else "cpu")

class Encoder(nn.Module):
    def __init__(self,input_size=300, hidden_size=75, num_layers=3, bidirectional=True,dropout=0.2):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, input):
        # seq_len, batch, input_dim = input.shape
        o, _ = self.gru(input) # o of shape (seq_len, batch, num_directions * hidden_size)
        o = self.dropout(o)
        return o

class QuestionPassageCoAttention(nn.Module):
    def __init__(self, hidden_size=75, batch_size=32):  
        super(QuestionPassageCoAttention, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.WQ_sub_u = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.WP_sub_u = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.WP_sub_v = torch.nn.Linear(hidden_size, hidden_size)
        self.Wg = torch.nn.Linear(4 * hidden_size, 4* hidden_size)
        self.Vt = torch.randn((self.batch_size, hidden_size, 1)).to(device)
        self.gatedRNN_fwd = nn.GRUCell(input_size= 4*hidden_size, hidden_size=hidden_size)
        self.gatedRNN_rev = nn.GRUCell(input_size= 4*hidden_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        
    
    def forward(self, quesEnc, passEnc):
        """
        lQ: length of question
        lP: length of question
        hiddenP: hidden size (output features size or size of last dim) of passage encoding (available here as passEnc) from the 'Encoding' Layer
        hiddenQ: hidden size (output features size or size of last dim) of question encoding (available here as quesEnc) from the 'Encoding' Layer i.e.
        bP: batch size received for the passEnc
        bQ: batch size received for the quesEnc
        
        """
        lP, bP, hiddenP = passEnc.shape
        lQ, bQ, hiddenQ = quesEnc.shape
        
        prev_vP_fwd = torch.randn((bP, self.hidden_size)).to(device) # (b,h)
        prev_vP_rev = torch.randn((bP, self.hidden_size)).to(device) # (b,h)
        vP_all_timesteps = torch.zeros(lP, bP, 2*self.hidden_size).to(device) # (lP,b,2h)
        
        
        # save quesEnc as shape with batch first i.e. (lQ, b, 2h) -> (b, lQ, 2h) for bmm operations
        quesEnc_bf = quesEnc.permute(1,0,2)

        # Over each time-step of the passage encoding
        for t,t_rev in zip(range(lP), reversed(range(lP))):
            # - Calculate scores (s_t in the paper; look at eq(4) in the paper) for all question words at this time step of the passage
            # - The process is as follows:
            # - At each time step of the passage, use:
            #   question encoding of all question words (quesEnc[j]), project it to dim of size h using a weight matrix WQ_sub_u,
            #   the passage encoding of curr time step (passEnc[t]), project it to dim of size h using a weight matrix WP_sub_u, 
            #   and previous hidden representation from the gated-attention based RNN (prev_vP), project it to dim of size h using a weight matrix WP_sub_v,
            # - Add all of the above up and add non-linearity (tanh) over the result
            # - shape of temp below should be (lQ, bQ, h)
            temp_fwd = torch.tanh( self.WQ_sub_u(quesEnc) + self.WP_sub_u(passEnc[t]) + self.WP_sub_v(prev_vP_fwd))
            temp_rev = torch.tanh( self.WQ_sub_u(quesEnc) + self.WP_sub_u(passEnc[t_rev]) + self.WP_sub_v(prev_vP_rev))
            # - Next calc uses bmm (batch multiply) operation on temp which requires batch dim first, so transform temp's dimensions
            temp_fwd = temp_fwd.permute([1,0,2])
            temp_rev = temp_rev.permute([1,0,2])
            # - Now temp's shape should be (bQ, lQ, h)
            # - For each entity in the batch for temp (i.e. ignore the batch dim for a second here), 
            #   to convert (lQ,h) part to a size (lQ, 1) (i.e. scalar scores over all question words), 
            #   by rules of matrix multilication, we need a matrix of size (h,1) which is self.Vt
            # - Batch multiply temp and Vt
            #   The result, s_t, would have a shape of (b, lQ, 1) where (lQ,1) represents the scalar score 
            #   for each question word
            s_t_fwd = torch.bmm(temp_fwd, self.Vt)
            s_t_rev = torch.bmm(temp_rev, self.Vt)
            # Compute softmax over dim 1 so that scores sum up to 1 over all question words. These are the attention-scores
            a_t_fwd = F.softmax(s_t_fwd, dim=1)
            a_t_rev = F.softmax(s_t_rev, dim=1)
            # - We need to do bmm (a_t, quesEnc) to calculate the context vector which is fed into the Bidirectional RNN for this layer
            # - a_t of shape (b, lQ, 1) needs to be changed to batch first (b, 1, lQ)
            # a_t : (b,lQ,1) -> squeeze(dim=2) -> (b,lQ) unsqueeze(dim=1) -> (b,1,lQ)
            a_t_fwd = a_t_fwd.squeeze(dim=2).unsqueeze(dim=1)
            a_t_rev = a_t_rev.squeeze(dim=2).unsqueeze(dim=1)
            
            c_t_fwd = torch.bmm(a_t_fwd,quesEnc_bf) # c_t is of shape (b,l,2h)
            c_t_rev = torch.bmm(a_t_rev,quesEnc_bf) # c_t is of shape (b,l,2h)
            # - Compute the sigmoid over cocatenated passEnc[t],c_t
            c_t_concat_fwd = torch.cat((passEnc[t].unsqueeze(1), c_t_fwd), dim=2) # c_t_concat is of shape (b,1,4h)
            c_t_concat_rev = torch.cat((passEnc[t_rev].unsqueeze(1), c_t_rev), dim=2) # c_t_concat is of shape (b,1,4h)

            g_t_fwd = torch.sigmoid(self.Wg(c_t_concat_fwd)) # g_t is of shape (b,1,4h)
            g_t_rev = torch.sigmoid(self.Wg(c_t_concat_rev)) # g_t is of shape (b,1,4h)
            # - Compute element-wise multiplication of g_t and c_t_concat and 
            #   use it as the input to the bidirectional RNN for time step t
            c_t_concat_star_fwd = torch.mul(g_t_fwd, c_t_concat_fwd).squeeze(1) # c_t_concat_star is of shape (b,4h)
            c_t_concat_star_rev = torch.mul(g_t_rev, c_t_concat_rev).squeeze(1) # c_t_concat_star is of shape (b,4h)

            prev_vP_fwd = self.gatedRNN_fwd(c_t_concat_star_fwd, prev_vP_fwd) # prev_vP_fwd shape(b,h)
            prev_vP_rev = self.gatedRNN_rev(c_t_concat_star_rev, prev_vP_rev) # prev_vP_fwd shape(b,h)

            vP_all_timesteps[t] = torch.cat((prev_vP_fwd, prev_vP_rev), dim=1)
            
            del temp_fwd, temp_rev, s_t_fwd,s_t_rev, a_t_fwd,a_t_rev, c_t_concat_fwd,c_t_concat_rev, g_t_fwd,g_t_rev, c_t_concat_star_fwd, c_t_concat_star_rev
        vP_all_timesteps = self.dropout(vP_all_timesteps)
        del quesEnc_bf, prev_vP_fwd, prev_vP_rev
        return vP_all_timesteps        


class SelfMatchingAttention(nn.Module):
    def __init__(self, hidden_size=75, batch_size=32):
        super(SelfMatchingAttention, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.WP_sub_v = torch.nn.Linear(2*hidden_size, hidden_size)
        self.WPhat_sub_v = torch.nn.Linear(2*hidden_size, hidden_size)
        self.Wg = torch.nn.Linear(4 * hidden_size, 4* hidden_size)
        self.Vt = torch.randn((self.batch_size, hidden_size, 1)).to(device)
        self.gatedRNN_fwd = nn.GRUCell(input_size= 4*hidden_size, hidden_size=hidden_size)
        self.gatedRNN_rev = nn.GRUCell(input_size= 4*hidden_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        
    
    def forward(self, passEnc):
        """
        lQ: length of question
        lP: length of question
        hiddenP: hidden size (output features size or size of last dim) of passage encoding (available here as passEnc) from the 'Encoding' Layer
        hiddenQ: hidden size (output features size or size of last dim) of question encoding (available here as quesEnc) from the 'Encoding' Layer i.e.
        bP: batch size received for the passEnc
        bQ: batch size received for the quesEnc
        
        """
        lP, bP, _ = passEnc.shape
        
        prev_hP_fwd = torch.randn((bP, self.hidden_size)).to(device) # (b,h)
        prev_hP_rev = torch.randn((bP, self.hidden_size)).to(device) # (b,h)
        hP_all_timesteps = torch.zeros(lP, bP, 2*self.hidden_size).to(device) # (lP,b,2h)
        
        # save quesEnc as shape with batch first i.e. (lQ, b, 2h) -> (b, lQ, 2h) for bmm operations
        passEnc_bf = passEnc.permute(1,0,2)
        passEnc_rev = torch.flip(passEnc, dims=[0]) # order of passage seq reversed
        passEnc_rev_bf = passEnc_rev.permute(1,0,2)

        # Over each time-step of the passage encoding
        for t,t_rev in zip(range(lP), reversed(range(lP))):
            # - shape of temp below should be (lQ, bQ, h)
            temp_fwd = torch.tanh( self.WP_sub_v(passEnc) + self.WPhat_sub_v(passEnc[t]))
            # THIS LINE IS CHANGED TO THE FOLLOWING LINE: temp_rev = torch.tanh( self.WP_sub_v(passEnc) + self.WPhat_sub_v(passEnc[t_rev]))
            temp_rev = torch.tanh( self.WP_sub_v(passEnc_rev) + self.WPhat_sub_v(passEnc_rev[t]))
            # - Next calc uses bmm (batch multiply) operation on temp which requires batch dim first, so transform temp's dimensions
            temp_fwd = temp_fwd.permute([1,0,2])
            temp_rev = temp_rev.permute([1,0,2])
            # - Now temp's shape should be (bQ, lQ, 2h)
            # - For each entity in the batch for temp (i.e. ignore the batch dim for a second here), 
            #   to convert (lQ,2h) part to a size (lQ, 1) (i.e. scalar scores over all question words), 
            #   by rules of matrix multilication, we need a matrix of size (h,1) which is Vt
            # - Batch multiply temp and Vt. 
            #   The result, s_t, would have a shape of (b, lQ, 1) where (lQ,1) represents the scalar score 
            #   for each question word
            s_t_fwd = torch.bmm(temp_fwd, self.Vt)
            s_t_rev = torch.bmm(temp_rev, self.Vt)
            
            # Compute softmax over dim 1 so that scores sum up to 1 over all question words. These are the attention-scores
            a_t_fwd = F.softmax(s_t_fwd, dim=1)
            a_t_rev = F.softmax(s_t_rev, dim=1)
            
            # - We need to do bmm (a_t, quesEnc) to calculate the context vector which is fed into the Bidirectional RNN for this layer
            # - a_t of shape (b, lQ, 1) needs to be changed to batch first (b, 1, lQ)
            # a_t : (b,lQ,1) -> squeeze(dim=2) -> (b,lQ) unsqueeze(dim=1) -> (b,1,lQ)
            a_t_fwd = a_t_fwd.squeeze(dim=2).unsqueeze(dim=1)
            a_t_rev = a_t_rev.squeeze(dim=2).unsqueeze(dim=1)
            
            c_t_fwd = torch.bmm(a_t_fwd,passEnc_bf) # c_t is of shape (b,l,2h)
            # THIS LINE IS CHANGED TO THE FOLLOWING LINE: c_t_rev = torch.bmm(a_t_rev,passEnc_bf) # c_t is of shape (b,l,2h)
            c_t_rev = torch.bmm(a_t_rev,passEnc_rev_bf) # c_t is of shape (b,l,2h)
            
            # - Compute the sigmoid over cocatenated passEnc[t],c_t
            c_t_concat_fwd = torch.cat((passEnc[t].unsqueeze(1), c_t_fwd), dim=2) # c_t_concat is of shape (b,1,4h)
            # THIS LINE IS CHANGED TO THE FOLLOWING LINE:  c_t_concat_rev = torch.cat((passEnc[t_rev].unsqueeze(1), c_t_rev), dim=2) # c_t_concat is of shape (b,1,4h)
            c_t_concat_rev = torch.cat((passEnc_rev[t].unsqueeze(1), c_t_rev), dim=2) # c_t_concat is of shape (b,1,4h)

            g_t_fwd = torch.sigmoid(self.Wg(c_t_concat_fwd)) # g_t is of shape (b,1,4h)
            g_t_rev = torch.sigmoid(self.Wg(c_t_concat_rev)) # g_t is of shape (b,1,4h)
            # - Compute element-wise multiplication of g_t and c_t_concat and 
            #   use it as the input to the bidirectional RNN for time step t
            c_t_concat_star_fwd = torch.mul(g_t_fwd, c_t_concat_fwd).squeeze(1) # c_t_concat_star is of shape (b,4h)
            c_t_concat_star_rev = torch.mul(g_t_rev, c_t_concat_rev).squeeze(1) # c_t_concat_star is of shape (b,4h)
            prev_hP_fwd = self.gatedRNN_fwd(c_t_concat_star_fwd, prev_hP_fwd) # prev_vP_fwd shape(b,h)
            prev_hP_rev = self.gatedRNN_rev(c_t_concat_star_rev, prev_hP_rev) # prev_vP_fwd shape(b,h)
            hP_all_timesteps[t] = torch.cat((prev_hP_fwd, prev_hP_rev), dim=1)
            del temp_fwd, temp_rev, s_t_fwd,s_t_rev, a_t_fwd,a_t_rev, c_t_concat_fwd,c_t_concat_rev, g_t_fwd,g_t_rev, c_t_concat_star_fwd, c_t_concat_star_rev
        hP_all_timesteps = self.dropout(hP_all_timesteps)
        # del passEnc_bf, prev_hP_fwd, prev_hP_bwd, passEnc_rev
        del passEnc_bf, passEnc_rev, prev_hP_fwd
        return hP_all_timesteps        
        
class AnswerPointerNetwork(nn.Module):
    def __init__(self,hidden_size=75,batch_size=32):
        super(AnswerPointerNetwork, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.WP_sub_h = nn.Linear(2*self.hidden_size,self.hidden_size)
        self.Wa_sub_h = nn.Linear(2*self.hidden_size,self.hidden_size)
        self.Vt1 = torch.randn((self.batch_size, self.hidden_size, 1)).to(device) # shape (b,h,1)
        self.Vt2 = torch.randn((self.batch_size, self.hidden_size, 1)).to(device)
        self.WQ_sub_u = nn.Linear(2*self.hidden_size,self.hidden_size)
        self.WQ_sub_v = nn.Linear(self.hidden_size,self.hidden_size)
        self.VQ_sub_r = torch.randn((1,1,self.hidden_size)).to(device)
        self.answerRNN = nn.GRUCell(input_size=2*hidden_size, hidden_size=2*hidden_size)
    
    def forward(self, passEnc, quesEnc):
        lP,bP,_ = passEnc.shape
        lQ,_,_ = quesEnc.shape
        assert bP == self.batch_size
        
        # Calculate question vector rQ which serves as initial hidden state for Answer RNN
        temp = torch.tanh(self.WQ_sub_u(quesEnc) + self.WQ_sub_v(self.VQ_sub_r)).permute([1,0,2]) # temp shape - (b,lQ,h)
        sQ = torch.bmm(temp, self.Vt1) # shape (b,lQ,1)
        a = F.softmax(sQ,dim=1) # shape (b,lQ,1)
        quesEnc_bf = quesEnc.permute([1,0,2]) # bf: batch_first; shape (b,lQ,2h)
        passEnc_bf = passEnc.permute([1,0,2]) # shape (b,lP,2h)
        # 'a' shape (b,lQ,1); quesEnc_bf shape (b,lQ,2h) not compatible for bmm; make 'a' compatible using permute
        a = a.permute([0,2,1]).contiguous() # shape (b,1,lQ)
        rQ = torch.bmm(a, quesEnc_bf).squeeze(1) # shape (b,1,2h) -> squeeze(1) -> (b,2h)
        # rQ serves as initial hidden state for the answer rnn

        answer_pointers = []
        # for answer_pos in range(2):
        #     temp2 = torch.tanh( self.WP_sub_h(passEnc) + self.Wa_sub_h(rQ)).permute([1,0,2]) # shape (b,lP,h)
        #     sP = torch.bmm(temp2, self.Vt2) # sP shape (b,lP,1)
        #     aP =  F.softmax(sP,dim=1).squeeze(2) # aP shape (b,lP)
        #     answer_pointers.append(aP)
        #     aP = aP.unsqueeze(1) # shape (b,1,lP)
        #     ct = torch.bmm(aP, passEnc_bf).squeeze(1) # bmm input shapes ( (b,1,lP) , (b,lP,2h) ) ; output shape (b,1,2h) -> squeeze(1) -> (b,2h)
        #     rQ = self.answerRNN(rQ, ct)

        temp2 = torch.tanh( self.WP_sub_h(passEnc) + self.Wa_sub_h(rQ)).permute([1,0,2]) # shape (b,lP,h)
        sP = torch.bmm(temp2, self.Vt2) # sP shape (b,lP,1)
        aP =  F.softmax(sP,dim=1).squeeze(2) # aP shape (b,lP)
        answer_pointers.append(aP)
        aP = aP.unsqueeze(1) # shape (b,1,lP)
        ct = torch.bmm(aP, passEnc_bf).squeeze(1) # bmm input shapes ( (b,1,lP) , (b,lP,2h) ) ; output shape (b,1,2h) -> squeeze(1) -> (b,2h)
        rQ = self.answerRNN(rQ, ct)
        temp2 = torch.tanh( self.WP_sub_h(passEnc) + self.Wa_sub_h(rQ)).permute([1,0,2]) # shape (b,lP,h)
        sP = torch.bmm(temp2, self.Vt2) # sP shape (b,lP,1)
        aP =  F.softmax(sP,dim=1).squeeze(2) # aP shape (b,lP)
        answer_pointers.append(aP)
        aP = aP.unsqueeze(1) # shape (b,1,lP)
        # ct = torch.bmm(aP, passEnc_bf).squeeze(1) # bmm input shapes ( (b,1,lP) , (b,lP,2h) ) ; output shape (b,1,2h) -> squeeze(1) -> (b,2h)
        # rQ = self.answerRNN(rQ, ct)
        
        del temp,sP,aP,ct,rQ,sQ,a,quesEnc_bf,passEnc_bf
        return tuple(answer_pointers)

class RNetModel(nn.Module):
    def __init__(self, batch_size=None, hidden_size=75,word_emb_dim=300):
        super(RNetModel, self).__init__()
        if batch_size is None:
            batch_size = base_config.get('training_batch_size')
        self.encoder = Encoder(input_size=word_emb_dim, hidden_size=hidden_size,num_layers=3, bidirectional=True,dropout=0.2)
        self.QP_CoAttention = QuestionPassageCoAttention(hidden_size=hidden_size, batch_size=batch_size)
        self.selfMatchingAttention = SelfMatchingAttention(hidden_size=hidden_size, batch_size=batch_size)
        self.answerPointer = AnswerPointerNetwork(hidden_size=hidden_size, batch_size=batch_size)
    
    def forward(self, passEmbs, quesEmbs):
        # Move input to GPU, if available
        passEmbs,quesEmbs = passEmbs.to(device), quesEmbs.to(device)
        uP = self.encoder(passEmbs)
        uQ = self.encoder(quesEmbs)
        vP = self.QP_CoAttention(uQ, uP)
        hP = self.selfMatchingAttention(vP)
        return self.answerPointer(hP, uQ)