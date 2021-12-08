import copy
import numpy as np
import torch

NUM_GABORS = 30
WIDTH = 128
HEIGHT = 128
NUM_FRAMES = 5
NUM_MEAN_ORIS = 4

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class GaborSequenceGenerator(torch.utils.data.DataLoader):
    def __init__(self, batch_size, num_trials, mode="reg", blank=False, 
                 roll=False, p_E = 0.1, e_pos="U", num_seq=4, num_frames=NUM_FRAMES, 
                 num_gabors=NUM_GABORS, width=WIDTH, height=HEIGHT, sigma_base=50, 
                 kappa=50, lam=1, gamma=0.2, seed=None, device="cpu", 
                 worker_init_fn=worker_init_fn):
        
        ### DATASET SHOULD BE A SEPARATE OBJECT
        super().__init__(self, batch_size=batch_size, worker_init_fn=worker_init_fn)

        # self.batch_size     = batch_size
        self.num_trials     = num_trials # number of images in a sequence (A, B, C, etc.)??
        self.__next_trial__ = 0
        self.mode           = mode
        self.blank          = blank
        self.roll           = roll
        self.p_E            = p_E
        self.e_pos          = e_pos
        
        self.sigma_base     = sigma_base # base gabor patch size
        self.kappa          = kappa
        self.lam            = lam
        self.gamma          = gamma
        self.num_seq        = num_seq # sequence length (number of images in a sequence)
        self.num_frames     = num_frames # duration for each image
        self.num_gabors     = num_gabors
        self.width          = width
        self.height         = height
        
        self.device         = device
        self.seed           = seed
        self.worker_init_fn = worker_init_fn
        
        self.prev_seq       = []

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        
        # initialize x/y positions and sizes for Gabors for each trial type
        self.trial_types = ["A", "B", "C", "D", "U", "X"]
        self.trial_type_idx = {trial_type: i for i, trial_type in enumerate(self.trial_types)}
        self.param_idx = {param_type: i for i, param_type in enumerate(["xpos", "ypos", "size"])}
 
        self.gabor_info = torch.rand(size=(len(self.trial_type_idx), len(self.param_idx), self.num_gabors))

        pos_idxs = (self.param_idx["xpos"], self.param_idx["ypos"])        
        self.gabor_info[:, pos_idxs] = 0.8 * (self.gabor_info[:, pos_idxs] * 2 - 1) # adjust position values
        self.gabor_info[:, self.param_idx["size"]] += 1 # adjust size values

        if self.e_pos == "D":
            self.gabor_info[self.trial_type_idx["U"]] = self.gabor_info[self.trial_type_idx["D"]]
        
        self.gabor_info[self.trial_type_idx["X"]] = 0 # blank should be all zeros



    def generate_batch(self):
        
        # GENERATE BATCH SEQUENCES
        sequence_trials = ["A", "B", "C", "D"]
        if self.blank:
            sequence_trials += "X"
        n_trials = len(sequence_trials)

        # batch size x nbr 
        batch_sequences = np.tile(
            [self.trial_type_idx[trial_type] for trial_type in sequence_trials], 
            reps=self.batch_size
            ).reshape(self.batch_size, n_trials)

        # replace some Ds with Es
        if self.mode == "surp":
            set_surp = np.where(np.random.rand(self.batch_size) < self.p_E)[0]
            batch_sequences[set_surp, 3] = self.trial_type_idx["U"]
        
        if self.roll:
            roll_shifts = np.random.randint(0, n_trials, size=self.batch_size) # select shifts
            row_idxs, col_idxs = np.ogrid[ : self.batch_size, : n_trials]
            col_idxs = col_idxs - roll_shifts[:, np.newaxis] # shift column indexing
            batch_sequences = batch_sequences[row_idxs, col_idxs] # reindex

        # cut sequences down
        batch_sequences = batch_sequences[:, : self.num_seq]
        

        # GENERATE ORIENTATIONS
        ori_means = np.random.randint(NUM_MEAN_ORIS, size=self.batch_size) * np.pi / NUM_MEAN_ORIS

        # Update orientation means for individual trials (N x B)
        all_ori_means = np.repeat(ori_means[np.newaxis, :], self.num_seq, 0)

        # Adjust mean orientation when A appears partway through sequence
        if self.roll:
            # Identify trials for which to update theta
            batch_sequences_no_surp = copy.deepcopy(batch_sequences)
            batch_sequences_no_surp[batch_sequences == self.trial_type_idx["U"]] = self.trial_type_idx["D"]
            min_val = np.arange(self.num_seq).reshape(1, -1)
            reset_ori_means = np.where(batch_sequences_no_surp < min_val)

            # create new theta values from new mean orientations
            new_ori_means = np.random.randint(NUM_MEAN_ORIS, size=self.batch_size) * np.pi / NUM_MEAN_ORIS # new thetas for each possible sequence
            all_ori_means[reset_ori_means[::-1]] = new_ori_means[reset_ori_means[0]] # update consistently within sequences

        else:
            A_starts = np.where(batch_sequences[0] == self.trial_type_idx["A"])[0][0]
            if A_starts > 0:
                new_ori_means = np.random.randint(NUM_MEAN_ORIS, size=self.batch_size) * np.pi / NUM_MEAN_ORIS # new thetas for each sequence
                all_ori_means[:, A_starts :] = new_ori_means[:, np.newaxis] # update from As

        # Update mean orientation for Es at the end of a sequence
        update_last_Es_only = True
        if self.mode == "surp":
            if update_last_Es_only:
                E_positions = np.where(batch_sequences[:, -1] == self.trial_type_idx["U"])[0]
                all_ori_means[-1, E_positions] += np.pi / 2
            else: # update ALL Es
                E_positions = np.where(batch_sequences == self.trial_type_idx["U"])
                all_ori_means[E_positions[::-1]] += np.pi / 2

        # Generate individual gabor orientations (P x N x B)
        theta = torch.FloatTensor(
            np.random.vonmises(
                mu=np.repeat(all_ori_means[np.newaxis, :, :], self.num_gabors, 0), 
                kappa= self.kappa)
            ) % np.pi # set upper bound as PI


        
    # python test_gabors.py --dataset gabors --batch_size 10 --img_dim 128 --train_what ft --epochs 25 --surprise_epoch 5 --pred_step 1 --print_freq 1 --roll --seq_len 5 --num_seq 4 --p_E 0.1 --gpu 0


        # DRAW GABORS (nbr patches x sequence length x batch size)
        xpos = self.gabor_info[batch_sequences, self.param_idx["xpos"]]
        ypos = self.gabor_info[batch_sequences, self.param_idx["ypos"]]
        sigma = self.gabor_info[batch_sequences, self.param_idx["size"]] * self.sigma_base # sizes

       # Setup mesh of coordinates to generate gabor patches from (W x H)
        X, Y  = torch.meshgrid(
            (torch.linspace(-1, 1, self.width), 
            torch.linspace(-1, 1, self.height))
            )
        
        # Create singleton patch and sequence dimension (W x H x nbr patches x sequence length),
        # and recenter based on patch location
        X = X.reshape(-1, 1, 1, 1) - xpos
        Y = Y.reshape(-1, 1, 1, 1) - ypos
        
        import pdb
        pdb.set_trace()

        # Rotate coordinates (W x H x P x N x B)
        x_theta =  X * theta.cos() + Y * theta.sin()
        y_theta = -X * theta.sin() + Y * theta.cos()
        
        # Create gabor patches ([B x W x H x P x N] * [P x N] broadcast)
        G = torch.exp(-((x_theta.pow(2) + self.gamma.pow(2) * y_theta.pow(2)) / (2 * sigma.pow(2)))) * torch.cos(2 * np.pi * x_theta / self.lam)
        
        
        # Reorder dimensions (B x N x W x H x P)
        G = G.permute(4, 3, 0, 1, 2)

        # Sum across patch dimension to collapse all patches into one frame (B x N x W x H)
        G = G.sum(dim=-1)
        # Find location of X in sequence and replace with blank frame        
        if "X" in seq:
            ii = [ix for ix,s in enumerate(seq) if s == "X"][0]
            G[:, ii] = torch.zeros(self.batch_size, self.width, self.height)
        # Create singleton Frame dimension
        G = G.unsqueeze(2)
        # Create singleton Channel dimension
        G = G.unsqueeze(2)
        # Repeat across frame and channel dimensions (B x N x C x F x W x H)
        G = G.repeat(1, 1, 3, self.num_frames, 1, 1)
        print("G")
        print(G.shape)
        return G
    
    #def generate_batch():
        
    #    generate_block
        
            
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
