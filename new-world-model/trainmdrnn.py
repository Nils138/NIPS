# -*- coding: utf-8 -*-
"""
Training routine for the MDRNN including a self-attention mechanism and corresponding penalty
"""
# MXNET and Gluon Packages
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from mxnet.ndarray.linalg import gemm2

# Project Intern Imports
from utils.EarlyStopping import EarlyStopping
from models.mdrnn import MDRNN
from models.sa_mdrnn import SA_MDRNN
from models.vae import VAE
from models.lstm import LSTM

# Additional Imports
import numpy as np
from matplotlib import pyplot as plt
import random
from os.path import join, exists
from os import mkdir
from data.loaders import RolloutObservationDataset as Dataset
import pickle


def create_data(game_length, batch_size, test_size,n_latents, n_actions,use_random_data=False,generate_data=True):
    """
    The function either returns a random dataset for the latent and action 
    training and test data or loads the random rollout from the neurosmash environment 
        game_length    : the maximum length of a single game
        batch_size     : the batch_size
        n_latents      : the number of latent nodes
        n_actions      : the number of possible actions
        use_random_data: boolean whether random training data is returned
        generate_data  : boolean whether new data are generated or old data are loaded from file
    """
    if use_random_data:
        #Create random input batchsize * num_latents * sequence_length 
        latent_train_data = nd.random.uniform(shape=(game_length, batch_size, n_latents))
        action_train_data = nd.random.uniform(shape=(game_length, batch_size, n_actions))
        reward_train_data = nd.random.uniform(shape=(game_length,batch_size))
        terminal_train_data = nd.random.uniform(shape=(game_length,batch_size))
        latent_test_data = nd.random.uniform(shape=(game_length, test_size, n_latents))
        action_test_data = nd.random.uniform(shape=(game_length, test_size, n_actions))
        reward_test_data = nd.random.uniform(shape=(game_length,test_size))
        terminal_test_data = nd.random.uniform(shape=(game_length,test_size))
        train_data = {"actions": action_train_data, "latents": latent_train_data, "rewards": reward_train_data, "terminals": terminal_train_data}
        test_data = {"actions": action_test_data, "latents": latent_test_data, "rewards": reward_test_data, "terminals": terminal_test_data}
        
    else:
        if generate_data:
            indices = np.arange(0,1000)
            train_idx = np.random.choice(indices,batch_size)
            #Choose random rollouts which are not used for training
            test_idx  = np.random.choice(indices[indices!=train_idx][0],test_size)
            train_data = load_dataset(train_idx, game_length, n_actions, n_latents)
            test_data = load_dataset(test_idx, game_length, n_actions, n_latents)
            
            with open('data/rnn_train', 'wb') as f:
                pickle.dump(train_data,f)
            
            with open('data/rnn_test', 'wb') as f:
                pickle.dump(test_data,f)
        else:
            with open('data/rnn_train', 'rb') as f:
                train_data = pickle.load(f)
            with open('data/rnn_test', 'rb') as f:
                test_data = pickle.load(f) 
                
            if not train_data['latents'].shape[1] is batch_size:
                print("Different batch size while loading previously generated data!!!")
    return train_data, test_data

def load_dataset(samples, game_length, n_actions, n_latents):
    """
    Gets the latent vectors from the VAE and returns them together with the reformed actions
        samples:     Array with indices of the data samples
        game_length: The maximum game length 
        n_actions:   Number of actions
        n_latents:   Number of latents
    """
    action_data   = nd.zeros(shape=(game_length,len(samples),n_actions))
    latent_data   = nd.zeros(shape=(game_length,len(samples),n_latents))
    reward_data   = nd.zeros(shape=(game_length,len(samples)))
    terminal_data = nd.zeros(shape=(game_length,len(samples)))
    for i, batch_idx in enumerate(samples):
            print(f'[{i}/{len(samples)}]')
            f = join("training_data", 'rollout_{}.npz'.format(batch_idx))
            with np.load(f) as data:
                observations = nd.array(data['observations'])
                #Observations of size (game_length,3,64,64)
                observations = observations.reshape((-1,3,64,64))
                #Create actions data of size (game_length,batch_size,n_actions)
                actions = nd.array(data['actions'])
                actions =nd.one_hot(actions,3)
                #Get rewards
                rewards = nd.array(data['rewards'])
                #Get terminal states
                terminals = nd.array(data['terminals'])

                #The game length 
                cur_game_length = len(actions)
                
                #Compute a pass over the VAE
                _, mu, logsigma = vae(observations)
                sigma = logsigma.exp()
                eps = mx.nd.random.normal(shape=sigma.shape)
                #Compute latent vector z (game_length * n_latents)
                z = sigma * eps + mu
                   
                #Ensure that all games have the same length by including zeros at the end
                if len(actions)<game_length:
                    get_padding = lambda dim: nd.zeros(shape=(game_length -cur_game_length, dim))
                    actions = nd.concat(actions,get_padding(n_actions),dim=0)
                    z = nd.concat(z,get_padding(n_latents),dim=0)
                    terminals = nd.concat(terminals,nd.zeros(shape=(game_length-cur_game_length)),dim=0)
                    rewards   = nd.concat(rewards, nd.zeros(shape=(game_length-cur_game_length)), dim=0)
                action_data[:,i,:]=actions
                latent_data[:,i,:]=z
                reward_data[:,i]= rewards
                terminal_data [:,i]=terminals
    return {"actions": action_data, "latents": latent_data, "rewards": reward_data, "terminals": terminal_data}
    


def max_game_length(batch_size):
    """
    Computes the maximum game length in the batch
        batch_size: Size of the batch
    """
    max_length = 0
    for batch_idx in range(batch_size):
            f = join("training_data", 'rollout_{}.npz'.format(batch_idx))
            with np.load(f) as data:
                if len(data['actions'])>max_length:
                    max_length = len(data['actions'])
    return max_length

def detach(hidden):
    """
    The function has been taken from: #https://github.com/sookinoby/RRN-MxNet/blob/master/Test-rnn.ipynb
    It checks whether the hidden states are a list/tuple and depending on that detaches either every element of the hidden states or
    the hidden states itself. The function is necessary to prevent that the gradients of the hidden states are accidentally computed.
    It returns the detaches hidden states. 
        hidden: the hidden states
    """
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def train_RNN(epoch, train_dic, seq_length, batch_size, game_length):
    """
    Trains an MDRNN given the training data for a certain number of epochs and prints the loss for certain batches
        epochs:            the number of epochs for which the MDRNN should be trained
        seq_length:        the duration of a sequence of observation the lstm tries to predict
        batch_size:        the size of the batches used to train the network
        game_length:       the maximum length of a game
        train_dic:         A dictionary contraining:
                            latents   = [seq_length, batch_size, n_latents]
                            actions   = [seq_length, batch_size, n_actions]
                            rewards   = [seq_length, batch_size]
                            terminals = [seq_length, batch_size]
    """
    total_loss = 0
    #Avoid learning across sequences
    hidden = model.lstm.begin_state(func= mx.nd.zeros, batch_size=batch_size, ctx=ctx)
    latents = train_dic['latents']
    actions = train_dic['actions']
    
    train_data = nd.concat(latents, actions, dim=-1)
    
    for i in range(0,game_length-1, seq_length):
        min_seq_length = min(seq_length,game_length-1-i)
        data = train_data[i:i+min_seq_length,:,:]
        target = latents[i+1 : i + min_seq_length + 1,:,:]
        hidden = detach(hidden)
        with autograd.record():
            if type(model)==MDRNN:
                mus, sigmas, logpi, states, rs, ts = model(data,hidden)
                L = model.loss(train_dic,(i+1, i + min_seq_length + 1), mus, sigmas, logpi, rs, ts)
            elif type(model)==SA_MDRNN:
                #print(data.shape)
                mus, sigmas, logpi, states, rs, ts, att = model(data,hidden)
                L = model.loss(train_dic,(i+1, i + min_seq_length + 1), mus, sigmas, logpi, rs, ts)
                pen = gemm2(att, att, transpose_b = True)
                # Penalty
                tmp = nd.dot(att[0], att[0].T) - nd.array(np.identity(att[0].shape[0]), ctx = ctx)
                pen = nd.sum(nd.multiply(nd.abs(tmp), nd.abs(tmp)))
                
                L = L + .05 * pen
                
            else:   
                pred , states = model(data,hidden)
                pred = nd.reshape(pred, shape=(min_seq_length, batch_size, -1))
                L = model.loss(pred,target) 
        L.backward()            
        trainer.step(batch_size,ignore_stale_grad=True)
        total_loss += mx.nd.sum(L).asscalar()                  
    print(f'Epoch {epoch} loss={total_loss/batch_size}')
    return total_loss/batch_size
            
            
def test_RNN(epoch, game_length, test_size,test_dic):
    """
    Tests the performance of the model
    epoch:      The training epoch
    game_length:The maximum game length
    test_size:  The size of the test dataset
    test_dic:   A dictionary contraining:
                latents   = [seq_length, batch_size, n_latents]
                actions   = [seq_length, batch_size, n_actions]
                rewards   = [seq_length, batch_size]
                terminals = [seq_length, batch_size]
    """
    
    latents = test_dic['latents']
    actions = test_dic['actions']
    #Avoid learning across sequences
    hidden = model.lstm.begin_state(func= mx.nd.zeros, batch_size=test_size, ctx=ctx)
    test_data = nd.concat(latents, actions, dim=-1)
    total_loss=0
    for i in range(0,game_length-1, seq_length):
        min_seq_length = min(seq_length,game_length-1-i)
        data = test_data[i:i+min_seq_length,:,:]
        target = test_data[i+1 : i + min_seq_length + 1,:,:]
        hidden = detach(hidden)
        if type(model)==MDRNN:
            mus, sigmas, logpi, states, rs, ts = model(data,hidden)
            L=model.loss(test_dic,(i+1, i + min_seq_length + 1), mus, sigmas, logpi, rs, ts)
        elif type(model)==SA_MDRNN:
            mus, sigmas, logpi, states, rs, ts, att = model(data,hidden)
            L=model.loss(test_dic,(i+1, i + min_seq_length + 1), mus, sigmas, logpi, rs, ts)
            pen = gemm2(att, att, transpose_b = True)
            # Penalty
            tmp = nd.dot(att[0], att[0].T) - nd.array(np.identity(att[0].shape[0]), ctx = ctx)
            pen = nd.sum(nd.multiply(nd.abs(tmp), nd.abs(tmp)))
            L = L + .05 * pen 
        else:   
            pred , states = model(data,hidden)
            target = nd.reshape(target, shape=(-1,n_latents))
            L = model.loss(pred,target) 
        total_loss += mx.nd.sum(L).asscalar()
    print(f'Epoch {epoch} test loss={total_loss/test_size}')
            
    return total_loss/test_size
                 
if __name__ == '__main__':
    generate_data = False #seet to false if you want to load a set with the same parameters
    batch_size =100
    test_size = 50
    game_length = max_game_length(999)
    seq_length = 60
    epochs = 100
    n_latents = 32
    n_actions = 3
    n_hiddens = 256
    n_gaussians = 5
    n_att1 = 10
    n_att2 = 5
    
    #which RNN to use: LSTM (F-F), MDRNN (T-F) or SA-MDRNN (T-T)
    gmm_layer = True
    att_layer = True
   
    #Use GPU if existent
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    
    #Create VAE
    vae = VAE(3, n_latents) 

    #Load parameters from file
    vae_file = join(join('logs', 'vae'), 'best.params')
    vae.load_parameters(vae_file, ctx=ctx)
    
    if gmm_layer:
        if att_layer:
            #Initialize SA_MDRNN
            model = SA_MDRNN(n_latents, n_actions, n_hiddens, n_gaussians, n_att1, n_att2)
        else:
            #Initialize MDRNN
            model = MDRNN(n_latents, n_actions, n_hiddens, n_gaussians) 
    else:
        #Initialize LSTM
        model = LSTM(n_latents,n_actions,n_hiddens)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    
    
    #Initialize trainer and early stopping
    trainer=gluon.Trainer(model.collect_params(),'adam')
    earlystopping = EarlyStopping('min', patience=20)
    
    # Load datasets
    train_data, test_data = create_data(game_length, batch_size,test_size, n_latents, n_actions, generate_data=generate_data)
    
    # Load existing model
    if gmm_layer:
        if att_layer:
            rnn_dir = join('logs','sa_mdrnn')
        else:
            rnn_dir = join('logs', 'mdrnn')
    else:
        rnn_dir = join('logs','lstm')
    if not exists(rnn_dir):
        mkdir(rnn_dir)
    
    
    #Logging the parameters which resulted in the best performance
    best_filename = join(rnn_dir, 'best.params')
    filename = join(rnn_dir, 'checkpoint.params')
    cur_best = None
    
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    for epoch in range(epochs):
        #Train MDRNN
        train_loss[epoch] = train_RNN(epoch, train_data, seq_length, batch_size, game_length)
        test_loss[epoch] = test_RNN(epoch, game_length, test_size, test_data)
        #earlystopping.step(test_loss)
        
        if not cur_best or test_loss[epoch] < cur_best:
            cur_best = test_loss[epoch]
            model.save_parameters(best_filename)    
        else:
            model.save_parameters(filename) 
    
    #Plot loss
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.xlabel('epochs')
    
    # if earlystopping.stop:
    #     print("End of Training because of early stopping at epoch {}".format(epoch))
    #     break
    
    

        

