# -*- coding: utf-8 -*-
"""
This file contains an MDRNN with an additional self-attention layer following Lin et al., 2017. 
In addition to the MDRNN, this model contains two additional dense layers computing the attention coefficients,
which are combined the hidden activation through matrix multiplication.
"""

#MXNET and Gluon packages
from mxnet import nd, gluon 
import mxnet as mx
from mxnet.gluon import nn, rnn
from mxnet.ndarray.linalg import gemm2
#from gluonts.mx.distribution import Gaussian

#Additional packages
import numpy as np

class SA_MDRNN(nn.Block):
    def __init__(self, n_latents, n_actions, n_hiddens, n_gaussians,n_w1,n_w2):
        super(SA_MDRNN,self).__init__()
        
        self.n_latents = n_latents
        self.n_gaussians = n_gaussians
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        self.d = n_w1
        self.r = n_w2
        
        #Create lstm layer 
        self.lstm = rnn.LSTMCell(self.n_hiddens, input_size = self.n_latents+self.n_actions)
        
        #Create MDN layers
        self.pi = nn.Dense(self.n_gaussians, in_units=self.n_hiddens*self.r) #changed output to n_gaussians
        self.mu = nn.Dense(self.n_gaussians*self.n_latents, in_units=self.n_hiddens*self.r) #changed to n_latents
        self.sigma = nn.Dense(self.n_gaussians*self.n_latents, in_units=self.n_hiddens*self.r)
        
        #Create reward state and terminal state layer
        self.rsts = nn.Dense(2, in_units = self.n_hiddens*self.r)
        
        #Create self-attention layers
        self.att1 = nn.Dense(self.d, in_units=self.n_hiddens, use_bias=False)
        self.att2 = nn.Dense(self.r, in_units=self.d, use_bias=False)
        
        
    def forward(self, latents_t, hidden):
        """
        latents_t: A mxnet ndarray which consists of the latent vector learned by the VAE concatinated with the actions
        hidden:  The state of the hidden layer used by the lstm. It is given to the forward pass to be able to start at a specific position
        
        The function computes a forward pass over the network and returns the learned variables:
            mus    = means, shape=[sequence_len, batch_size, n_gaussians, n_latents]
            sigmas = the standard deviation, shape=[sequence_len, batch_size, n_gaussians, n_latents]
            logpi  = the logarithm of pi, shape=[sequence_len, batch_size, n_gaussians]
            states = the current state of the hidden layer, shape=[1,batch_size,n_hiddens]
            terminal_state = whether the state is a terminal state or not
            reward_state   = the reward to be expected in a state
            att = self-attention coefficients
        """
        #Get the sequence length and batchsize 
        sequence_len, batch_size = latents_t.shape[0], latents_t.shape[1]
        
        #Compute the output of the lstm layer
        outputs, states = self.lstm.unroll(length = sequence_len, inputs = latents_t, layout = 'TNC', merge_outputs=True, begin_state=hidden)#valid length for padding (batch size) 
        
        #Pass every timestep through attention layer
        for k in range(sequence_len):
            _outputs = outputs[k,:,:]
            _outputs = nd.expand_dims(_outputs, axis=0)
            #Flatten output of lstm 
            _output_sub = _outputs.reshape((-1, self.n_hiddens))
            
            #Pass through attention MLP layers
            _w = nd.tanh(self.att1(_output_sub))
            w = self.att2(_w)
            _att = w.reshape((-1, batch_size, self.r)) # Timestep * Batch * r
            att = nd.softmax(_att, axis = 0)
            
            att = att.reshape((batch_size, -1, self.r))
            _outputs = _outputs.reshape((batch_size, -1, self.n_hiddens))
            
            #matrix multiplication to get r*n_hidden matrix with attention correlates
            _att_matrix = gemm2(att, _outputs, transpose_a = True)
            _att_matrix = nd.expand_dims(_att_matrix, axis=0)
            if k==0:
                att_output = _att_matrix
            else:
                att_output = nd.concat(att_output, _att_matrix, dim=0)

        outputs = att_output.reshape((-1,self.n_hiddens*self.r))
        
        #Compute the mean values
        mus = self.mu(outputs)
        mus = nd.reshape(mus, shape=(sequence_len, batch_size, self.n_gaussians, self.n_latents))
        
        #Compute standard deviations
        sigmas = self.sigma(outputs)
        sigmas = nd.reshape(sigmas, shape=(sequence_len, batch_size, self.n_gaussians, self.n_latents))
        sigmas = nd.exp(sigmas)

        #Compute pi 
        pi = self.pi(outputs)
        pi = nd.reshape(pi, shape=(sequence_len, batch_size, self.n_gaussians))
        logpi = nd.log_softmax(pi, axis=-1)
        
        #Compute terminal and reward state
        params = self.rsts(outputs)
        params = nd.reshape(params, shape=(sequence_len, batch_size, 2))
        reward_state, terminal_state = params[:,:,0], params[:,:,1] 
        

        return mus, sigmas, logpi, states, terminal_state, reward_state, att
    
    def gmm_loss(self,batch, mus, sigmas, logpi, reduce=True):
        """
        Calculates the gmm loss
        mus   : ndarray of the means, shape=[sequence_len, batch_size, n_gaussians, n_latents]
        sigmas: ndarray of the standard deviations, shape=[sequence_len, batch_size, n_gaussians, n_latents]
        logpi : ndarray containing the logarithm of the pi values, shape=[sequence_len, batch_size, n_gaussians]
        reduce: boolean defining whether the negative mean of the log should be returned
        """        
        
        # # gluontslog prob
        batch = nd.reshape(batch, shape=(batch.shape[0],-1, 1, batch.shape[-1]))
        normal_dist = Gaussian(mus, sigmas)
        g_log_probs = normal_dist.log_prob(batch)
        
        ## same from here on
        g_log_probs = logpi + nd.sum(g_log_probs, axis=-1)
        max_log_probs = nd.max(g_log_probs, axis=-1, keepdims=True)#[0]
        g_log_probs = g_log_probs - max_log_probs
         	
        g_probs = nd.exp(g_log_probs)
        probs = nd.sum(g_probs, axis=-1)
        log_probs = nd.squeeze(max_log_probs) + nd.log(probs)
        
        if reduce:
            return - nd.mean(log_probs)
        return - log_probs
    
    
    def loss(self, data_dic, idx, mus, sigmas, logpi, rs, ts):
        """
        Calculate the loss using the information of whether a terminal state was reached, the reward gained
        and the gmm loss
        mus:        ndarray of the means, shape=[sequence_len, batch_size, n_gaussians, n_latents]
        sigmas:     ndarray of the standard deviations, shape=[sequence_len, batch_size, n_gaussians, n_latents]
        logpi:      ndarray containing the logarithm of the pi values, shape=[sequence_len, batch_size, n_gaussians]
        idx:        The idx of the current sequence of the form (start,end)
        data_dic:   A dictionary contraining:
                    latents   = [seq_length, batch_size, n_latents]
                    actions   = [seq_length, batch_size, n_actions]
                    rewards   = [seq_length, batch_size]
                    terminals = [seq_length, batch_size]
        """
        latents   = data_dic['latents']
        rewards   = data_dic['rewards']
        terminals = data_dic['terminals']
        gmm_loss = self.gmm_loss(latents[idx[0]:idx[1],:,:], mus, sigmas, logpi)
        ts_loss  = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        rs_loss = gluon.loss.L2Loss()
        l = (gmm_loss + ts_loss(ts,terminals[idx[0]:idx[1],:]) +rs_loss(rs,rewards[idx[0]:idx[1],:]))/(self.n_latents + 2)
        return l
        
         