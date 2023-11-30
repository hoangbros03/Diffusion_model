import math

import torch
from torch import nn
import pytorch_lightning as pl

from modules import *
class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.beta_small=1e-4
        self.beta_large=0.02
        self.t_range= t_range
        self.in_size = in_size
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Pre-defined 
        bilinear=True
        factor = 2

        self.inc = DoubleConv(img_depth,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256,128 //factor, bilinear)
        self.up3 = Up(128,64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128,8)

        self.save_hyperparameters()

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0/(
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)
    
    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t,128,16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256,4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128,8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64,16)
        x = self.up3(x ,x1) + self.pos_encoding(t,64,32)
        output = self.outc(x)
        return output
    
    def beta(self,t):
        # t bigger -> beta bigger
        return self.beta_small + (t/self.t_range)* (self.beta_large - self.beta_small)
    
    def alpha(self, t):
        # t bigger -> alpha smaller
        return 1- self.beta(t)
    
    def alpha_bar(self,t):
        # Decrease over time
        return math.prod([self.alpha(j) for j in range(t)])
    
    def get_loss(self, batch):
        # Randomly get the step number in each image in the
        # current batch
        ts = torch.randint(0, self.t_range, [batch.shape[0]]).to(self.device)
        noise_imgs = []

        # Get rand with shape like batch shape
        epsilons = torch.randn(batch.shape, device=self.device)
        
        # Create noise images for each image in the current batch
        for i in range(len(ts)):
            # Get acumulate alpha
            a_hat = self.alpha_bar(ts[i])

            # Append a noise image at i-th time step into noise_imgs
            # t inc -> a_hat dec -> (1-a_hat)*noise inc -> ok
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1- a_hat) * epsilons[i])
            )
            
        # Stack it to make a matrix like input matrix
        noise_imgs = torch.stack(noise_imgs, dim=0).to(self.device)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        
        return loss
    
    def denoise_sample(self,x ,t):
        with torch.no_grad():
            if t>1:
                z = torch.randn(x.shape)
            else:
                z =0
            e_hat = self.forward(x, t.view(1,1).repeat(x.shape[0],1))
            pre_scale=1/math.sqrt(self.alpha(t))         
            e_scale = (1-self.alpha(t)) / math.sqrt(1- self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t))*z
            x = pre_scale*(x- e_scale*e_hat) + post_sigma
            return x
        
    def training_step(self, batch):
        loss = self.get_loss(batch)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch):
        loss = self.get_loss(batch)
        self.log("val/loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer