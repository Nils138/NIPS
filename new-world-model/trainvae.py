""" Training VAE """
from os.path import join, exists
from os import mkdir

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.loss import L2Loss
from mxnet.gluon.data.vision import transforms

from models.vae import VAE

from utils.EarlyStopping import EarlyStopping
from data.loaders import RolloutObservationDataset as Dataset


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    loss = L2Loss()
    BCE = loss(recon_x, x)
    KLD = -0.5 * nd.sum(1 + 2 * logsigma - mu ** (2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch, train_loader, model, batch_size):
    """ One training epoch """
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        with autograd.record():
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss[0]
        mx_trainer.step(batch_size)
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader),
                loss[0].asnumpy() / len(data)))

    print('====> Epoch: {} Average loss: {}'.format(
        epoch, train_loss.asnumpy() / len(train_loader)))


def test(test_loader, model):
    """ One test epoch """
    test_loss = 0
    for data in test_loader:
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar)[0] 
    test_loss /= len(test_loader)
    print('====> Test set loss: {}'.format(test_loss.asnumpy()))
    return test_loss
    

if __name__ == '__main__':
    # Hyper parameters
    batch_size = 32
    LSIZE = 32
    n_epochs = 1000
    noreload = False
    nosamples = True
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()
    
    # Load datasets
    dataset_train = Dataset('training_data', transform_train, train=True)
    dataset_test = Dataset('training_data', transform_test, train=False)
    
    train_loader = mx.gluon.data.DataLoader(
         dataset_train, batch_size, shuffle=True, num_workers=0)   
    test_loader = mx.gluon.data.DataLoader(
         dataset_test, batch_size, shuffle=True, num_workers=0)
    
    # Initialize model
    model = VAE(3, LSIZE)
    model.collect_params().initialize(ctx=mx.cpu())
    mx_trainer = gluon.Trainer(model.collect_params(), 'adam')
    earlystopping = EarlyStopping('min', patience=10)
    
    # Make missing directories
    vae_dir = join('logs', 'vae')
    if not exists(vae_dir):
        mkdir(vae_dir)
        mkdir(join(vae_dir, 'samples'))
    
    # Load existing model
    reload_file = join(vae_dir, 'best.params')
    if not noreload and exists(reload_file):
        model.load_parameters(reload_file, ctx=mx.cpu())
        print("\nReloaded model parameters from {}".format(reload_file))
    
    cur_best = None
    
    # Iterate over epochs
    for epoch in range(1, n_epochs + 1):
        train(epoch, train_loader, model, batch_size)
        test_loss = test(test_loader, model)
        earlystopping.step(test_loss)
    
        # Checkpointing
        best_filename = join(vae_dir, 'best.params')
        filename = join(vae_dir, 'checkpoint.params')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
            model.save_parameters(best_filename)    
        else:
            model.save_parameters(filename)
    
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
