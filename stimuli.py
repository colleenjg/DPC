import numpy as np
import torch
from math import pi

class GaborSequenceGenerator(object):
    def __init__(self, batch_size, num_trials, mode='reg',
                 NUM_FRAMES=5, NUM_GABORS=30, WIDTH=128, HEIGHT=128,
                 sigma_base = 50, kappa = 50, lam = 1, gamma=0.2,
                 seed=1000, device='cpu'):
        
        self.batch_size     = batch_size
        self.num_trials     = num_trials
        self.__next_trial__ = 0
        self.mode           = mode
        
        self.sigma_base     = sigma_base
        self.kappa          = kappa
        self.lam            = lam
        self.gamma          = gamma
        self.NUM_FRAMES     = NUM_FRAMES
        self.NUM_GABORS     = NUM_GABORS
        self.WIDTH          = WIDTH
        self.HEIGHT         = HEIGHT
        
        self.device         = device
        self.seed           = seed
        
        self.prev_seq       = []
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        from math import pi
        self.gabor_info = {'A' : {},
                           'B' : {},
                           'C' : {},
                           'D' : {},
                           'E' : {}}
        
        for trial in self.gabor_info.keys():
            self.gabor_info[trial]['xpos'] = 0.8 * (torch.rand(size=(self.NUM_GABORS,))*2 - 1)
            self.gabor_info[trial]['ypos'] = 0.8 * (torch.rand(size=(self.NUM_GABORS,))*2 - 1)
            self.gabor_info[trial]['size'] = 1.0 +  torch.rand(size=(self.NUM_GABORS,))
        
    def generate_batch(self):
        ori_mean = np.random.randint(4, size=self.batch_size) * pi/4
        
        if self.mode == 'reg':
            seq = ['A', 'B', 'C', 'D']
        elif self.mode == 'surp':
            seq = ['A', 'B', 'C']
            seq += ['D'] if np.random.rand() <= 0.9 else ['E']
        
        self.prev_seq.append(seq)
        
        X, Y  = torch.meshgrid((torch.linspace(-1, 1, self.WIDTH), torch.linspace(-1, 1, self.HEIGHT)))
        X     = X.unsqueeze(-1).unsqueeze(-1)
        Y     = Y.unsqueeze(-1).unsqueeze(-1)
#         X     = (X.unsqueeze(-1)*torch.ones(1, self.batch_size * self.num_frames)).permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
#         Y     = (Y.unsqueeze(-1)*torch.ones(1, self.batch_size * self.num_frames)).permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)

        theta = torch.Tensor(np.random.vonmises(mu=ori_mean * np.ones((self.NUM_GABORS, len(seq), self.batch_size)), kappa= self.kappa))
        if 'E' in seq:
            theta[:, -1] = (theta[:, -1] + pi/2) % pi
        
        xpos    = torch.stack([self.gabor_info[trial]['xpos'] for trial in seq]).permute(1, 0)
        ypos    = torch.stack([self.gabor_info[trial]['ypos'] for trial in seq]).permute(1, 0)
        
        X       = (X - xpos).unsqueeze(-1)
        Y       = (Y - ypos).unsqueeze(-1)
        
        sigma   = torch.stack([self.sigma_base / self.gabor_info[trial]['size'] for trial in seq]).permute(1, 0)

        x_theta =  X*theta.cos() + Y*theta.sin()
        y_theta = -X*theta.sin() + Y*theta.cos()

        x_theta = x_theta.permute(4, 0, 1, 2, 3)
        y_theta = y_theta.permute(4, 0, 1, 2, 3)
        
        G = torch.exp(-((x_theta.pow(2) + self.gamma * y_theta.pow(2))/2*sigma**2))*torch.sin(2*pi*x_theta/self.lam)
        G = G.permute(0, 4, 1, 2, 3)
        G = G.sum(dim=-1)
        
        G = G.unsqueeze(2)
        G = G.unsqueeze(2)

        G = G.repeat(1, 1, self.NUM_FRAMES, 3, 1, 1)
            
        return G
            
    def __getitem__(self, ix):
        if ix < self.__len__():
            return self.generate_batch()
        else:
            raise IndexError
    
    def __len__(self):
        return self.num_trials
    
    def __next__(self):
        if self.__next_trial__ < self.__len__():
            self.__next_trial__ += 1
            return self.generate_batch()
        else:
            raise StopIteration
    
    def _set_mode(self, mode):
        self.mode = mode
