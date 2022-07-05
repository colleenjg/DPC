import copy
import logging
import warnings

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils import data
import torchvision

from dataset import augmentations


NUM_GABORS = 30
GAB_IMG_LEN = 1
NUM_MEAN_ORIS = 8
U_ADJ = 90

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def get_num_classes():
    raise NotImplementedError(
        "Supervised learning not implemented for Gabors dataset."
        )

#############################################
class GaborSequenceGenerator(data.Dataset):
    def __init__(
        self, 
        width=256, 
        height=256, 
        mode="train",
        dataset_name="Gabors",
        transform=None,
        train_len=5000, # training dataset length
        seq_len=10,
        num_seq=5,
        num_gabors=NUM_GABORS, 
        num_mean_oris=NUM_MEAN_ORIS,
        gab_img_len=GAB_IMG_LEN,
        same_possizes=True,
        gray=True, 
        roll=False, 
        shift_frames=False,
        unexp=True,
        U_prob=0.1, 
        diff_U_possizes=False, 
        size_ran=[10, 20], # degrees
        deg_width=120, # image width, in degrees (if curved)
        ori_std_dev=0.25, # radians
        num_cycles=1, # number of visible cycles (approx.)
        psi=0.25, # phase (prop of cycle)
        gamma=1, # aspect ratio
        return_label=False,
        supervised=False,
        seed=None, # used only if same_possizes is True
        ):
        
        self.mode         = mode
        self.dataset_name = dataset_name
        self.supervised   = supervised

        self.width           = int(width) # pixels
        self.height          = int(height) # pixels
        self.seq_len         = int(seq_len) # nbr frames per seq
        self.num_seq         = int(num_seq) # nbr of sequences to sample
        self.num_gabors      = int(num_gabors) # nbr of Gabor patches
        self.num_mean_oris   = int(num_mean_oris) # nbr of mean orientations
        self.gab_img_len     = int(gab_img_len) # nbr frames per Gabor image

        if self.mode == "train":
            self.dataset_len = int(train_len)
        else:
            self.dataset_len = int(train_len / 2)

        self.full_len        = int(seq_len * num_seq) # full sequence length
        if self.mode == "test" and self.supervised:
            self.full_len *= 5

        self.same_possizes   = bool(same_possizes)
        self.gray            = bool(gray)
        self.roll            = bool(roll)
        self.shift_frames    = bool(shift_frames)
        self.unexp           = bool(unexp)
        self.diff_U_possizes = bool(diff_U_possizes)
        
        self.size_ran           = size_ran # base gabor patch size range
        self.kappa              = 1.0 / (ori_std_dev ** 2)
        self.psi_rad            = 2 * psi * np.pi
        self.gamma              = gamma
        self.num_sigmas_visible = 3
                
        self.prev_seq        = []
        
        # initialize relevant properties
        self.set_num_cycles_adj(num_cycles)
        self.set_deg_per_pix(deg_width)
        self.set_size_ran(size_ran)
        self.set_U_prob(U_prob)
        self.set_transform(transform)
        self.num_base_seq_repeats
        self.mean_oris
        self.image_type_dict

        if self.same_possizes:
            self.set_possizes(seed=seed)

        if supervised:
            return_label = True
        self.return_label = return_label


    def set_num_cycles_adj(self, num_cycles=1):

        if num_cycles <= 0:
            raise ValueError("num_cycles must be strictly positive.")

        if num_cycles < 0.5 or num_cycles > 3:
            raise NotImplementedError(
                "The Gabor generation code is best suited for 1 cycle to be "
                "visible. The approximations used work reasonably well between "
                "0.5 and 3 cycles, but not beyond.")

        self.num_cycles = num_cycles

        fact = 1.2 ** (num_cycles - 1)
        num_cycles = num_cycles * fact
        self.num_cycles_adj = num_cycles


    def set_deg_per_pix(self, deg_width=120, curved=True):
        """
        self.set_deg_per_pix()
        """
        
        if curved:
            self.deg_per_pix = deg_width / self.width
        else:
            dist = self.width / (2 * np.tan(np.deg2rad(deg_width / 2)))
            self.deg_per_pix = np.rad2deg(np.arctan(1 / dist))


    def set_size_ran(self, size_ran=[10, 20]):
        """
        self.set_size_ran()
        """

        if len(size_ran) != 2:
            raise ValueError("'size_ran' should have length 2.")
        if size_ran[0] <= 0 or size_ran[1] <= 0:
            raise ValueError("'size_ran' values must be strictly positive.")
        if size_ran[0] > size_ran[1]:
            size_ran = [size_ran[1], size_ran[0]]

        self.size_ran = size_ran


    def set_U_prob(self, U_prob=0):
        """
        self.set_U_prob()
        """
        
        if U_prob < 0 or U_prob > 1:
            raise ValueError("U_prob must be between 0 and 1, inclusively.")
        self.U_prob = U_prob


    def set_transform(self, transform=None):
        """
        set_transform()
        """

        if transform is None:
            self.transform = torchvision.transforms.Compose([
                augmentations.ToTensor(),
                augmentations.Normalize(),
            ])
        else:
            self.transform = transform


    @property
    def mean_oris(self):
        """
        self.mean_oris
        """
        
        if not hasattr(self, "_mean_oris"):
            self._mean_oris = np.arange(0, 360, 360 / self.num_mean_oris)
        return self._mean_oris


    @property
    def mean_oris_U(self):
        """
        self.mean_oris_U
        """
        
        if not hasattr(self, "_mean_oris_U"):
            self._mean_oris_U = (self.mean_oris + U_ADJ) % 360 
        return self._mean_oris_U


    @property
    def all_mean_oris(self):
        """
        self.all_mean_oris
        """
        
        if not hasattr(self, "_all_mean_oris"):
            if self.U_prob > 0:
                mean_oris = np.concatenate([self.mean_oris, self.mean_oris_U])
            else:
                mean_oris = self.mean_oris
            self._all_mean_oris = np.unique(mean_oris)
        return self._all_mean_oris


    @property
    def base_seq(self):
        """
        self.base_seq
        """
        
        if not hasattr(self, "_base_seq"):
            self._base_seq = ["A", "B", "C", "D"]
            if self.gray:
                self._base_seq.append("G")
        return self._base_seq


    @property
    def num_images_total(self):
        """
        self.num_images_total
        """
        
        if not hasattr(self, "_num_images_total"):
            self._num_images_total = int(
                np.ceil(self.full_len / self.gab_img_len)
                ) + int(self.roll) # add a buffer of 1, if rolling
        return self._num_images_total


    @property
    def num_base_seq_repeats(self):
        """
        self.num_base_seq_repeats
        """

        if not hasattr(self, "_num_base_seq_repeats"):
            self._num_base_seq_repeats = int(
                np.ceil(self.num_images_total / len(self.base_seq))
                )
        return self._num_base_seq_repeats


    @property
    def image_type_dict(self):
        """
        self.image_type_dict
        """

        if not hasattr(self, "_image_type_dict"):
            image_types = self.base_seq[:]
            if self.U_prob > 0:
                image_types.append("U")

            self._image_type_dict = {
                image_type: i for i, image_type in enumerate(image_types)
                }
            if not self.diff_U_possizes:
                self._image_type_dict["U"] = self._image_type_dict["D"]
        return self._image_type_dict


    @property
    def image_type_labels(self):
        """
        self.image_type_labels
        """

        if not hasattr(self, "_image_type_labels"):
            image_type_labels_dict = dict()
            for i, key in enumerate(self.image_type_dict.keys()):
                image_type_labels_dict[key] = i
            self._image_type_labels = image_type_labels_dict
        return self._image_type_labels
        

    @property
    def image_label_types(self):
        """
        self.image_label_types
        """
        
        if not hasattr(self, "_image_label_types"):
            image_label_types_dict = dict()
            for key, val in self.image_type_labels.items():
                image_label_types_dict[val] = key
            self._image_label_types = image_label_types_dict
        return self._image_label_types


    @property
    def positions(self):
        """
        self.positions: num_pos x 2 x num_gabors
        """
        if not hasattr(self, "_positions"):
            num_pos = max(self.image_type_dict.values()) + 1
            arr_size = (num_pos, 2, self.num_gabors)
            if hasattr(self, "_possize_rng"):
                positions = self._possize_rng.random(size=arr_size)
            else:
                positions = np.random.random(size=arr_size)
            positions = (positions - 0.5) * 0.9 + 0.5 # keep mostly within image
            if self.gray:
                gray_idx = self.image_type_dict["G"]
                positions[gray_idx] = np.nan
            if self.same_possizes: # don't store otherwise
                self._positions = positions
            return positions
        else:
            return self._positions


    @property
    def sizes(self):
        """
        self.sizes: num_pos x num_gabors
        """
        if not hasattr(self, "_sizes"):
            num_sizes = max(self.image_type_dict.values()) + 1
            arr_size = (num_sizes, self.num_gabors)
            if hasattr(self, "_possize_rng"):
                sizes = self._possize_rng.random(size=arr_size)
            else:
                sizes = np.random.random(size=arr_size)
            # adjust to degrees within the size range
            sizes = sizes * np.diff(self.size_ran)[0] + self.size_ran[0]
            # convert to pixels
            sizes = sizes / self.deg_per_pix
            if self.gray:
                gray_idx = self.image_type_dict["G"]
                sizes[gray_idx] = 0
            if self.same_possizes: # don't store otherwise
                self._sizes = sizes
            return sizes
        else:
            return self._sizes


    def get_sequence_idxs(self, get_num_seq=True):
        """
        self.get_sequence_idxs()

        self.sequence_idxs: num_seq x seq_len
        """
        
        if self.full_len < self.num_seq * self.seq_len:
            raise NotImplementedError(
                "self.full_len must be at least equal to "
                "self.num_seq x self.seq_len."
                )

        # identify the start point for each sequence (consecutive)
        if get_num_seq:
            seq_start_idx = self.seq_len * np.expand_dims(
                    np.arange(self.num_seq), -1
                ) 
        else:
            seq_start_idx = self.seq_len * np.expand_dims(
                    np.arange(self.full_len // self.seq_len), -1
                ) 

        # identify indices for each sequence (num_seq x seq_len)
        seq_idx = seq_start_idx + np.expand_dims(
            np.arange(self.seq_len), 0
            )
        
        self._sequence_idxs = torch.from_numpy(seq_idx)

        return self._sequence_idxs


    @property
    def sub_batch_idxs(self):
        """
        self.sub_batch_idxs
        """
        
        if not hasattr(self, "_sub_batch_idxs"):
            num_poss = self.full_len // self.seq_len - self.num_seq + 1
            step_size = self.num_seq // 2 # sub-batch of sequences overlap by half

            self._sub_batch_idxs = torch.vstack([
                torch.arange(i, i + self.num_seq) 
                for i in range(0, num_poss, step_size)
                ])

        return self._sub_batch_idxs


    def set_possizes(self, seed=None, reset=True):
        """
        self.set_possizes()
        """
        
        if seed is not None:
            self._possize_rng = np.random.RandomState(seed)
 
        for attr_name in ["_sizes", "_positions"]:
            if hasattr(self, attr_name) and reset:
                delattr(self, attr_name)                    
 
        self.sizes
        self.positions


    def sample_mean_ori(self):
        """
        self.sample_mean_ori()
        """

        mean_ori = np.random.choice(self.mean_oris)
        return mean_ori


    def image_to_image_idx(self, seq_images):
        """
        self.image_to_image_idx(seq_images)
        """

        seq_image_idxs = np.asarray(
            [self.image_type_dict[i] for i in seq_images]
            )

        return seq_image_idxs


    def image_to_image_label(self, seq_images):
        """
        self.image_to_image_label(seq_images)
        """

        seq_image_labels = np.asarray(
            [self.image_type_labels[i] for i in seq_images]
            )

        return seq_image_labels


    def image_label_to_image(self, seq_image_labels):
        """
        self.image_label_to_image(seq_image_labels)
        """
        
        seq_image_labels = [
            self.image_label_types[i] for i in seq_image_labels
        ]
        
        return seq_image_labels


    def _replace_Ds_with_Us(self, seq_images):
        """
        self._replace_Ds_with_Us(seq_images)
        """
        
        seq_images = seq_images[:]
        if not self.unexp or self.U_prob == 0:
            return seq_images
        
        num_Ds  = seq_images.count("D")
        if num_Ds == 0:
            return seq_images

        Us = np.random.rand(num_Ds) < self.U_prob 
        if Us.sum():
            U_idxs = np.where(np.asarray(seq_images) == "D")[0][Us]
            for U_idx in U_idxs:
                seq_images[U_idx] = "U"

        return seq_images


    def get_sequence_images(self):
        """
        self.get_sequence_images()
        """
        
        if self.roll:
            roll_shift = np.random.choice(len(self.base_seq))
            seq_images = np.roll(self.base_seq, roll_shift).tolist()
        else:
            seq_images = self.base_seq[:] # make a copy
        
        seq_images = seq_images * self.num_base_seq_repeats
        seq_images = seq_images[: self.num_images_total]

        seq_images = self._replace_Ds_with_Us(seq_images)

        return seq_images


    def get_image_mean_oris(self, seq_images):
        """
        self.get_image_mean_oris(seq_images)
        """

        image_idxs = self.image_to_image_idx(seq_images)
        min_image_idx = min(image_idxs)
        ori_reset_idxs = np.where(np.asarray(image_idxs) == min_image_idx)[0]
        
        image_mean_oris = []
        mean_ori = self.sample_mean_ori()
        for i, seq_image in enumerate(seq_images):
            if i in ori_reset_idxs:
                mean_ori = self.sample_mean_ori()
            ori = mean_ori
            if seq_image == "U":
                ori = (mean_ori + U_ADJ) % 360
            elif seq_image == "G":
                ori = np.nan
            image_mean_oris.append(ori)

        image_mean_oris = np.asarray(image_mean_oris)
        
        return image_mean_oris


    def get_all_oris(self, image_mean_oris):
        """
        self.get_all_oris(image_mean_oris)

        all_oris: [0, 360[
        """

        num_images = len(image_mean_oris)
        ori_dims = (num_images, self.num_gabors)
        
        all_ori_jitter_rad = np.random.vonmises(
            mu=0, kappa=self.kappa, size=ori_dims
            )
        all_ori_jitter = np.rad2deg(all_ori_jitter_rad)
        all_oris = np.asarray(image_mean_oris).reshape(num_images, 1)

        # catches NaN-related warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value", RuntimeWarning
                )
            all_oris = (all_oris + all_ori_jitter) % 360

        return all_oris
    

    def create_gabor(self, gabor_size, theta=0):
        """
        self.create_gabor(gabor_size)

        size (pixels)
        theta (degrees)
        """

        ksize = int(
            self.num_sigmas_visible * gabor_size * 1.0 / 
            (2 * np.sqrt(2 * np.log(2)))
            )
        lambd = ksize / self.num_cycles_adj
        sigma = ksize / (self.num_sigmas_visible * 2)

        ksize *= 3 # expand around Gabor to avoid artifacts

        gabor_filter = cv2.getGaborKernel(
            ksize=(ksize, ksize), # image size in pixels
            sigma=sigma, # gauss std in pixels
            theta=theta, # orientation in radians
            lambd=lambd, # wavelength in pixels
            gamma=self.gamma, # ratio
            psi=self.psi_rad, # phase in radians
            )
        
        return gabor_filter


    def overlay_gabors(self, image_gabor_patches, positions):
        """
        self.overlay_gabors(image_gabor_patches, positions)
        """

        image = np.zeros((self.height, self.width))

        if not np.isfinite(positions.any()):
            return image

        patch_sizes = np.asarray(
            [len(gabor_patch) for gabor_patch in image_gabor_patches]
            ).reshape(1, -1)
        positions = copy.deepcopy(positions)
        positions[0] *= self.height
        positions[1] *= self.width
        positions = positions.astype(int)

        # get start and end for each Gabor (in height and width dims)
        starts = positions - patch_sizes // 2        
        ends = starts + patch_sizes
        starts = np.clip(starts, a_min=0, a_max=None).astype(int)
        starts_gab = (patch_sizes - ends + starts).astype(int)
        ends[0] = np.clip(ends[0], a_min=0, a_max=self.height).astype(int)
        ends[1] = np.clip(ends[1], a_min=0, a_max=self.width).astype(int)
        ends_gab = (starts_gab + ends - starts).astype(int)

        # get min to max pixel values
        gab_max = max(
            [gabor_patch.max() for gabor_patch in image_gabor_patches]
            )
        gab_min = min(
            [gabor_patch.min() for gabor_patch in image_gabor_patches]
            )

        for g, gabor_patch in enumerate(image_gabor_patches):
            # define mask
            canvas_hei = slice(starts[0, g], ends[0, g])
            canvas_wid = slice(starts[1, g], ends[1, g])
            patch_hei = slice(starts_gab[0, g], ends_gab[0, g])
            patch_wid = slice(starts_gab[1, g], ends_gab[1, g])

            image[canvas_hei, canvas_wid] += gabor_patch[patch_hei, patch_wid]
        
        image = np.clip(image, a_min=gab_min, a_max=gab_max)

        # normalize to 0 to 1
        if image.max() == image.min():
            image[:] = 0.5
        else:
            max_val = max(np.absolute([image.min(), image.max()])) * 1.1
            image = (image + max_val) / (2 * max_val)

        # adjust to 0-255 range
        image = (image * 255).astype(np.uint8)

        return image
    

    def _generate_sequence(self):
        """
        self._generate_sequence()
        """

        # identify images and orientations for the full sequence        
        seq_images = self.get_sequence_images()
        seq_image_mean_oris = self.get_image_mean_oris(seq_images)

        # Get orientation per image and patch
        all_seq_oris = self.get_all_oris(seq_image_mean_oris)
        all_seq_oris = np.deg2rad(all_seq_oris) # convert to radians

        # Get sizes
        seq_image_idxs = self.image_to_image_idx(seq_images)
        sizes = self.sizes[seq_image_idxs]
        num_images, num_gabors = sizes.shape

        # Draw Gabors sequences
        gabor_patches =[
            self.create_gabor(size, ori) 
            for size, ori in zip(sizes.reshape(-1), all_seq_oris.reshape(-1))
            ]
        gabor_patches = [
            [gabor_patches[i * num_gabors + g] for g in range(num_gabors)] 
            for i in range(num_images)
            ]

        # Overlay Gabors
        positions = self.positions[seq_image_idxs]
        gabor_seq = np.asarray([
            self.overlay_gabors(image_gabor_patches, image_positions)
            for image_gabor_patches, image_positions 
            in zip(gabor_patches, positions)
        ]) # full_length x H x W

        # Repeat per channel
        gabor_seq = np.repeat(np.expand_dims(gabor_seq, axis=-1), 3, axis=-1)

        # Repeat frames per image
        if self.gab_img_len != 1:
            gabor_seq = np.repeat(gabor_seq, self.gab_img_len, axis=0)

        # Retrieve labels
        seq_image_labels = self.image_to_image_label(seq_images)
        seq_labels = np.vstack([seq_image_labels, seq_image_mean_oris]).T
        seq_labels = np.repeat(seq_labels, self.gab_img_len, axis=0)

        # Randomly shift the start point
        if self.roll and self.shift_frames and self.gab_img_len > 1:
            shift = np.random.choice(self.gab_img_len)
            gabor_seq = gabor_seq[shift : shift + self.full_len]
            seq_labels = seq_labels[shift : shift + self.full_len]
        
        return gabor_seq, seq_labels


    def generate_sequences(self):
        """
        self.generate_sequences()
        """

        gabor_seq, seq_labels = self._generate_sequence()

        gabor_seq = [
            Image.fromarray(gabor_img, mode="RGB") for gabor_img in gabor_seq
            ]
        
        if self.transform is not None:
            gabor_seq = self.transform(gabor_seq) # apply same transform
        else:
            gabor_seq = [
                torchvision.transforms.ToTensor()(gabor_img) 
                for gabor_img in gabor_seq
                ]

        gabor_seq = torch.stack(gabor_seq, 0)

         # Convert to torch 
        seq_labels = torch.from_numpy(seq_labels)

        # Cut into sequences
        if self.mode == "test" and self.supervised:
            get_num_seq = False
        else:
            get_num_seq = True

        seq_idx = self.get_sequence_idxs(get_num_seq=get_num_seq)
        gabor_seq = gabor_seq[seq_idx] # N_all x SL x C x H x W
        seq_labels = seq_labels[seq_idx]

        # Move color dimension
        gabor_seq = torch.moveaxis(gabor_seq, 2, 1)

        if self.mode == "test" and self.supervised:
            # SUB_B x N x C x SL x H x W
            gabor_seq = gabor_seq[self.sub_batch_idxs]
            seq_labels = seq_labels[self.sub_batch_idxs]

        return gabor_seq, seq_labels


    def __getitem__(self, ix):
        gabor_seq, seq_labels = self.generate_sequences()
        
        if self.return_label:
            return gabor_seq, seq_labels
        else:
            return gabor_seq

    
    def __len__(self):
        return self.dataset_len
        
