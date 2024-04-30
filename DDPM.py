import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        slope = (beta_T-beta_1)/(T-1)
        beta_t = beta_1+slope*(t_s-1)
        sqrt_beta_t = beta_t**0.5
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1/(alpha_t**0.5)
        # print(alpha_t_bar)
        if(t_s.shape):
          alpha_t_bar = torch.Tensor([1]).repeat(t_s.shape).to(device)
          for i in range(t_s.shape[0]):
            for tt in range(1,t_s[i]+1):
              b_t = beta_1+slope*(tt-1)
              a_t = 1 - b_t
              alpha_t_bar[i]*=a_t
        else:
          alpha_t_bar = torch.Tensor([1]).to(device)
          for tt in range(1,t_s+1):
            b_t = beta_1+slope*(tt-1)
            a_t = 1 - b_t
            alpha_t_bar*=a_t

        
        sqrt_alpha_bar = alpha_t_bar**0.5
        sqrt_oneminus_alpha_bar=(1-alpha_t_bar)**0.5

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  
        # print(conditions.shape,images.shape)
        cmv = self.dmconfig.condition_mask_value
        num_classes = self.dmconfig.num_classes
        mp=self.dmconfig.mask_p
        t=torch.randint(low=1,high=T+1,size=(conditions.shape[0],)).to(device)
        tt=t/T
        e=torch.normal(0, 1, size=(conditions.shape[0],1,28,28)).to(device)
        c = F.one_hot(conditions,num_classes)
        rr = torch.rand(conditions.shape[0])
        mask = rr<mp
        c[mask,:] = cmv
        schedule_dict = self.scheduler(t_s = torch.tensor(t).to(device))
        xt = schedule_dict['sqrt_alpha_bar'].reshape(conditions.shape[0],1,1,1).to(device)*images+\
        schedule_dict['sqrt_oneminus_alpha_bar'].reshape(conditions.shape[0],1,1,1).to(device)*e
        ec = self.network(xt,tt.reshape(conditions.shape[0],1,1,1),c)
        noise_loss = self.loss_fn(ec,e)



        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  
        # mp=self.dmconfig.mask_p
        # print(conditions.shape)
        cmv = self.dmconfig.condition_mask_value
        with torch.no_grad():
          X_t = torch.normal(0, 1, size=(conditions.shape[0],1,28,28)).to(device)      
          for t in range(T,0,-1):
            z=torch.zeros_like(X_t).to(device)
            if(t>1):
              z = torch.normal(0, 1, size=(conditions.shape[0], 1,28,28)).to(device)
            tt=t/T
            schedule_dict = self.scheduler(t_s = torch.tensor(t).to(device))
            c_n = cmv*torch.ones_like(conditions)
            t_t = tt*torch.ones(conditions.shape[0],1).to(device)
            ec = self.network(X_t,t_t,conditions)
            en = self.network(X_t,t_t,c_n)
            e_b = (1+omega)*ec - omega*en
            X_t = (schedule_dict['oneover_sqrt_alpha'].to(device)*(X_t - \
            (schedule_dict['beta_t'].to(device)/schedule_dict['sqrt_oneminus_alpha_bar'].to(device))*e_b))\
            + schedule_dict['sqrt_beta_t'].to(device)*z




        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images