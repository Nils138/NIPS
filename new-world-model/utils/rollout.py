###################################################################################
# The code from this file was taken from: https://github.com/ctallec/world-models
# and was adapted to make it compatible with the Neuromash environment
################################################################################

""" Various auxiliary utilities """
from os.path import join, exists
import torch
from torchvision import transforms
from models.sa_mdrnn import SA_MDRNN
from models.vae import VAE
from models.controller import Controller
import mxnet as mx
from mxnet import nd
from env import Neurosmash
import numpy as np


ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.params') for m in ['vae', 'sa_mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        self.vae = VAE(3, LSIZE)
        self.vae.load_parameters(vae_file, ctx=mx.cpu())

        self.mdrnn = SA_MDRNN(LSIZE, ASIZE, RSIZE, 5, 10, 5)
        self.mdrnn.load_parameters(rnn_file, ctx=mx.cpu())

        self.controller = Controller(LSIZE, RSIZE, ASIZE)

        # load controller if it was previously saved
        if exists(ctrl_file):
            self.controller.load_parameters(ctrl_file, ctx=mx.cpu())

        # Initialize environment
        ip = "127.0.0.1"
        port = 13000
        size = 64
        timescale = 37
        self.env = Neurosmash.Environment(ip, port, size, timescale)
        
        self.device = device
        self.time_limit = time_limit

    def get_action_and_transition(self, state, hidden):
        """ Get action and transition.

        Encode state to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action. Contains conversions from tensors to ndarrays 
        and back, to integrate the vae/rnn with the controller

        :args state: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        # VAE
        state_nd = nd.array(state)
        _, latent_mu, _ = self.vae(state_nd)
        
        # Controller
        mu_tensor = torch.tensor(latent_mu.asnumpy())
        action = self.controller(mu_tensor, hidden[0])
        
        #RNN
        action_nd = nd.array(action)
        hidden = [nd.array(hidden[i]) for i in range(2)]
        # Concatenate actions to latent vector
        latents = nd.concat(latent_mu, action_nd)
        latents = nd.reshape(latents, (1, 1, latents.shape[1]))
        _, _, _, next_hidden, _, _, _ = self.mdrnn(latents, hidden)
        
        # Convert back into a tensor
        next_hidden = [torch.tensor(hidden[i].asnumpy()) for i in range(2)]
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        end, reward, state = self.env.reset()
        
        hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]

        i = 0
        while True:
            # Transform state
            state = torch.reshape(torch.tensor(state, dtype=torch.float32), (64, 64, 3))
            state = transform(state).unsqueeze(0).to(self.device)
            
            # Get the reward
            actions, hidden = self.get_action_and_transition(state, hidden)
            action = np.argmax(actions)            
            end, reward, state = self.env.step(action)
            
            # End conditions
            if not end == 0 or i > self.time_limit:
                return - reward
            i += 1
            
