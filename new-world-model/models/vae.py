"""
Variational encoder model, used as a visual model
for our model of the world.
"""
from mxnet.gluon import nn
import mxnet as mx

class Decoder(nn.Block):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.fc1 = nn.Dense(1024, activation='relu')
        self.deconv1 = nn.Conv2DTranspose(128, 5, strides=2, activation='relu')
        self.deconv2 = nn.Conv2DTranspose(64, 5, strides=2, activation='relu')
        self.deconv3 = nn.Conv2DTranspose(32, 6, strides=2, activation='relu')
        self.deconv4 = nn.Conv2DTranspose(img_channels, 6, strides=2, 
                                          activation='sigmoid')

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))  # double unsqueeze
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        reconstruction = self.deconv4(x)
        return reconstruction

class Encoder(nn.Block):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2D(32, 4, strides=2, activation='relu')
        self.conv2 = nn.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = nn.Conv2D(128, 4, strides=2, activation='relu')
        self.conv4 = nn.Conv2D(256, 4, strides=2, activation='relu')

        self.fc_mu = nn.Dense(latent_size)
        self.fc_logsigma = nn.Dense(latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape((x.shape[0], -1))
        
        mu = self.fc_mu(x)        
        logsigma = self.fc_logsigma(x)
        return mu, logsigma

class VAE(nn.Block):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = mx.nd.random.normal(shape=sigma.shape)
        z = sigma * eps + mu

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
