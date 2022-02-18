# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 11:38:53 2020

@author: Ellen
"""
#Mxnet and Gluon Packages
from mxnet import nd, gluon
import mxnet as mx
from mxnet.gluon import nn, rnn

#Additional Packages
import numpy as np


class LSTM(nn.Block):
    def __init__(self, n_latents, n_actions, n_hiddens):
        super(LSTM,self).__init__()
        
        self.n_latents = n_latents
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        
        #Create lstm layer 
        self.lstm = rnn.LSTM(self.n_hiddens, input_size = self.n_latents+self.n_actions)   
        self.dense = nn.Dense(self.n_latents, in_units=self.n_hiddens)
        
        
    def forward(self, latents_t, hidden):#latents already has actions concatenated
        """
        latents_t: A mxnet ndarray which consists of the latent vector learned by the VAE concatinated with the actions
        hidden:  The state of the hidden layer used by the lstm. It is given to the forward pass to be able to start at a specific position
        
        The function computes a forward pass over the network and returns the learned variables:  
        
        """
        #Get the sequence length and batchsize 
        sequence_len, batch_size = latents_t.shape[0], latents_t.shape[1]
        
        #Compute the output of the lstm layer
        outputs, states = self.lstm(latents_t, hidden) 
        #Flatten output of lstm 
        outputs = outputs.reshape((-1,self.n_hiddens))
       
        pred = self.dense(outputs)
        
        return pred, states
    
    def loss(self, pred, target):
        #l = loss.L2Loss()
        l = gluon.loss.L2Loss()
        return  l(pred,target)
    
 
# class MaskedL2Loss(gluon.loss.L2Loss):
#     """The L2loss with masks."""
#     def forward(self, pred, label, game_length):
#         #weights = nd.expand_dims(nd.ones_like(label), axis=-1)
#         weights = nd.ones_like(label)
#         weights = nd.SequenceMask(weights, nd.array(game_length), True, axis=0)
#         print(super(MaskedL2Loss, self).forward(pred, label, weights))
#         return super(MaskedL2Loss, self).forward(pred, label, weights)