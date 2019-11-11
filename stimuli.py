import numpy as np
import torch
from math import pi

class GaborSequenceGenerator(object):
    def __init__(self, batch_size, num_trials, mode='reg', blank=False,
                 NUM_FRAMES=5, NUM_GABORS=30, WIDTH=128, HEIGHT=128,
                 sigma_base = 50, kappa = 50, lam = 1, gamma=0.2,
                 seed=1000, device='cpu'):
        
        self.batch_size     = batch_size
        self.num_trials     = num_trials
        self.__next_trial__ = 0
        self.mode           = mode
        self.blank          = blank
        
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
                           'E' : {},
                           'X' : {}} # placeholder, I know it's wasteful, but I ain't spending any more time on it
        
        for trial in self.gabor_info.keys():
            self.gabor_info[trial]['xpos'] = 0.8 * (torch.rand(size=(self.NUM_GABORS,))*2 - 1)
            self.gabor_info[trial]['ypos'] = 0.8 * (torch.rand(size=(self.NUM_GABORS,))*2 - 1)
            self.gabor_info[trial]['size'] = 1.0 +  torch.rand(size=(self.NUM_GABORS,))
        
    def generate_batch(self):
        
        # Generate regular or surprise sequence
        if self.blank == True:    
            if self.mode == 'reg':
                seq = ['A', 'B', 'C', 'D', 'X']
            elif self.mode == 'surp':
                seq = ['A', 'B', 'C']
                seq += ['D', 'X'] if np.random.rand() <= 0.9 else ['E', 'X']
        else:
            if self.mode == 'reg':
                seq = ['A', 'B', 'C', 'D']
            elif self.mode == 'surp':
                seq = ['A', 'B', 'C']
                seq += ['D'] if np.random.rand() <= 0.9 else ['E']
            
        # Shift sequence to random starting point and take 4 elements in sequence
        seq = list(np.roll(seq, np.random.randint(len(seq))))[:4]        
        # Save sequence
        self.prev_seq.append(seq)
        
        # Generate mean orientation
        ori_mean = np.random.randint(4, size=self.batch_size) * pi/4
        
        # Setup mesh of coordinates to generate gabor patches from (W x H)
        X, Y  = torch.meshgrid((torch.linspace(-1, 1, self.WIDTH), torch.linspace(-1, 1, self.HEIGHT)))
        # Create singleton patch and sequence dimension (W x H x P x N)
        X     = X.unsqueeze(-1).unsqueeze(-1)
        Y     = Y.unsqueeze(-1).unsqueeze(-1)
        
        # Get gabor patch locations (P x N)
        xpos    = torch.stack([self.gabor_info[trial]['xpos'] for trial in seq]).permute(1, 0)
        ypos    = torch.stack([self.gabor_info[trial]['ypos'] for trial in seq]).permute(1, 0)
        
        # Re-centre coordinates by patch locations ([W x H x P x N] + [P x N] broadcast) and add singleton batch dimension (W x H x P x N x B)
        X       = (X - xpos).unsqueeze(-1)
        Y       = (Y - ypos).unsqueeze(-1)
        
        # Generate gabor orientations (P x N x B)
        theta = torch.FloatTensor(np.random.vonmises(mu=ori_mean * np.ones((self.NUM_GABORS, len(seq), self.batch_size)), kappa= self.kappa))
        
        # Adjust if A comes late in sequence (new trial)
        if 'A' in seq[-2:]:
            theta[:, -2:] = (theta[:, -2:] + torch.randint(4, size=(self.batch_size,), dtype=torch.float32) * pi/4) % pi
        # Adjust if E is in sequence
        if 'E' in seq:
            ii = [ix for ix, s in enumerate(seq) if s=='E'][0] # Get index of E in sequence
            theta[:, ii] = (theta[:, ii] + pi/2) % pi          # Adjust E orientations
        
        # Rotate coordinates (W x H x P x N x B)
        x_theta =  X*theta.cos() + Y*theta.sin()
        y_theta = -X*theta.sin() + Y*theta.cos()

        # Re-order dimensions (B x W x H x P x N)
        x_theta = x_theta.permute(4, 0, 1, 2, 3)
        y_theta = y_theta.permute(4, 0, 1, 2, 3)
        
        # Generate patch sizes via sigma (P x N)
        sigma   = torch.stack([self.sigma_base / self.gabor_info[trial]['size'] for trial in seq]).permute(1, 0)
        
        # Create gabor patches ([B x W x H x P x N] * [P x N] broadcast)
        G = torch.exp(-((x_theta.pow(2) + self.gamma * y_theta.pow(2))/2*sigma**2))*torch.sin(2*pi*x_theta/self.lam)
        
        # Reorder dimensions (B x N x W x H x P)
        G = G.permute(0, 4, 1, 2, 3)
        # Sum across patch dimension to collapse all patches into one frame (B x N x W x H)
        G = G.sum(dim=-1)
        # Find location of X in sequence and replace with blank frame
        if 'X' in seq:
            ii = [ix for ix,s in enumerate(seq) if s == 'X'][0]
            G[:, ii] = torch.zeros(self.batch_size, self.WIDTH, self.HEIGHT)
        # Create singleton Frame dimension
        G = G.unsqueeze(2)
        # Create singleton Channel dimension
        G = G.unsqueeze(2)
        
        # Repeat across frame and channel dimensions (B x N x C x F x W x H)
        G = G.repeat(1, 1, 3, self.NUM_FRAMES, 1, 1)
            
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
