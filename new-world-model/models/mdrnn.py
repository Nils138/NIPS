#MXNET and Gluon packages
from mxnet import nd, gluon 
from mxnet.gluon import nn, rnn
#from gluonts.mx.distribution import Gaussian

#Additional packages
from scipy.special import *

class MDRNN(nn.Block):
    def __init__(self, n_latents, n_actions, n_hiddens, n_gaussians):
        super(MDRNN,self).__init__()
        
        self.n_latents = n_latents
        self.n_gaussians = n_gaussians
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        
        #Create lstm layer 
        self.lstm = rnn.LSTM(self.n_hiddens, input_size = self.n_latents+self.n_actions)
        
        #Create MDN layers
        self.pi = nn.Dense(self.n_gaussians, in_units=self.n_hiddens) #changed output to n_gaussians
        self.mu = nn.Dense(self.n_gaussians*self.n_latents, in_units=self.n_hiddens) #changed to n_latents
        self.sigma = nn.Dense(self.n_gaussians*self.n_latents, in_units=self.n_hiddens)
        #Reward state and terminal state layer
        self.rsts = nn.Dense(2, in_units = self.n_hiddens)
        #self.dense = nn.Dense(32, in_units=self.n_hiddens)
        
        
    def forward(self, latents_t, hidden):#latents already has actions concatenated
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
        
        """
        #Get the sequence length and batchsize 
        sequence_len, batch_size = latents_t.shape[0], latents_t.shape[1]
        
        #Compute the output of the lstm layer
        outputs, states = self.lstm(latents_t, hidden) 
        #Flatten output of lstm 
        outputs = outputs.reshape((-1,self.n_hiddens))
       
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
        

        return mus, sigmas, logpi, states, terminal_state, reward_state
    
    def gmm_loss(self,batch, mus, sigmas, logpi, reduce=True):
        """
        Calculates the gmm loss
        mus   : ndarray of the means, shape=[sequence_len, batch_size, n_gaussians, n_latents]
        sigmas: ndarray of the standard deviations, shape=[sequence_len, batch_size, n_gaussians, n_latents]
        logpi : ndarray containing the logarithm of the pi values, shape=[sequence_len, batch_size, n_gaussians]
        reduce: boolean defining whether the negative mean of the log should be returned
        """
        
        # g_log_probs = logpi + nd.sum(g_log_probs, axis=-1)
        # max_log_probs = nd.max(g_log_probs, axis=-1, keepdims=True)[0]
        # g_log_probs = g_log_probs - max_log_probs
     	
        # g_probs = nd.exp(g_log_probs)
        # probs = nd.sum(g_probs, axis=-1)
        # print(nd.mean(probs))
        # log_probs = nd.squeeze(max_log_probs) + nd.log(probs)
        
        
        # # gluonts version
        batch = nd.reshape(batch, shape=(batch.shape[0],-1, 1, batch.shape[-1]))
        normal_dist = Gaussian(mus, sigmas)
        g_log_probs = normal_dist.log_prob(batch)
        
        ## scipy version, stale gradients
        # batch = batch.asnumpy()
        # mus = mus.asnumpy()
        # sigmas = sigmas.asnumpy()
        # batch = np.expand_dims(batch,axis=-2)
        # normal_dist = norm(loc=mus, scale=sigmas)
        # g_log_probs = normal_dist.logpdf(batch)
        # g_log_probs = mx.nd.array(g_log_probs)
        
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
        
         