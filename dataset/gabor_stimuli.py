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
from utils import gabor_utils


NUM_GABORS = 30
GAB_IMG_LEN = 1


logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def check_if_is_gabor(dataset):
    """
    check_if_is_gabor(dataset)

    Returns whether the input dataset is a GaborSequenceGenerator object.

    Required args
    -------------
    - dataset : torch data.Dataset object
        Torch dataset object.

    Returns
    -------
    - is_gabor : bool
        Whether the input dataset is a GaborSequenceGenerator object.
    """

    if isinstance(dataset, str):
        raise ValueError("'dataset' should be a Gabor dataset object.")

    is_gabor = isinstance(dataset, GaborSequenceGenerator)

    return is_gabor


#############################################
class GaborSequenceGenerator(data.Dataset):
    """
    Class for generating Gabor sequences.

    Attributes
    ----------
    - all_mean_oris : 1D array
        All possible mean orientations (for expected or unexpected image types)
    - base_seq : list
        Base image type sequence (e.g., ["A", "B", "C", "D", "G"])
    - class_dict_decode : dict
        Dictionary for decoding class names from class labels
    - class_dict_encode : dict
        Dictionary for encoding class names into class labels
    - dataset_len : int
        Dataset length
    - dataset_name : str
        Dataset name
    - deg_per_pix : float
        Visual degrees to pixels conversion used.
    - diff_U_possizes : bool
        Whether U positions and sizes are different from D positions and sizes.
    - full_len : int
        Length of continuous sequences to generate (before breaking down into 
        individual sequences)
    - gab_img_len : int
        Number of frames to repeat for Gabor image
    - gamma : float
        Gabor patch spatial aspect ratio (width to height ratio)
    - gray : bool
        If True, grayscreen images are included in the base sequence.
    - height : int
        Individual Gabor image height (pixels)
    - image_type_dict : dict
        Dictionary for associating image types (e.g., "A", "B", etc.) to 
        indices for the positions and sizes arrays.
    - kappa : float 
        Orientation dispersion parameter: 1 / (std dev in rad) ^ 2
    - mean_oris : 1D array
        Mean orientations to sample from for expected image types.
    - mean_oris_U : 1D array
        Mean orientations to sample from for unexpected image types (i.e., "U").
    - mode : str
        Dataset mode (i.e., "train", "val", or "test") 
    - num_base_seq_repeats : int
        Number of base sequence repeats (minimum) needed to generate a full 
        continuous sequences.
    - num_cycles_adj : float
        Adjusted number cycles to use in drawing Gabor patches. 
    - num_frames_total : int
        Total number of frames per sequence
    - num_gabors : int
        Number of Gabor patches per image.
    - num_mean_oris : int
        Number of equally spaced orientations to sample from, between 0 and 360 
        degrees. 
    - num_seq : int
        Number of consecutive sequences to return per sample
    - num_sigmas_visible : float
        Number of standard deviations to make visible on each side of a Gabor 
        patch.
    - positions : 3D array
        Gabor patch positions for each image type, with 
        dims: image type x [height, width] x number of gabors
    - psi_rad : float
        Gabor patch phase (in radians)
    - return_label : bool
        If True, when sequences are sampled, labels for each image are returned 
        as well.
    - roll : bool
        If True, the base sequence can be rolled when generating a full 
        continuous sequence.
    - same_possizes : bool
        If True, positions and sizes are fixed for the dataset.
    - seq_len : int
        Number of frames per sequence.
    - shift_frames : bool
        If True and self.roll is True, rolling also allows shifting the start 
        point of a sequence down to the level of repeated frames.
    - size_ran : list
        Size range for the Gabor patches (in visual degrees).
    - sizes : 2D array
        Gabor patch sizes for each image type (in visual degrees), with 
        dims: image type x number of gabors
    - sub_batch_idxs : 2D Tensor
        Tensor for reindexing a sequences into overlapping sub-batches 
        (for "test" mode), with dims: number of sub batches x num_seq 
    - supervised : bool
        If True, dataset is set to supervised mode.
    - transform : torch Transform
        Transform to apply to the Gabor sequences sampled.
    - unexp : bool
        If True, dataset is set to produce unexpected sequences.
    - U_prob : float
        Probability of replacing each individual D with a U frame, if unexp is 
        True. 
    - width : int
        Individual Gabor image width (pixels).

    Methods
    -------
    - self.create_gabor(gabor_size)
        Creates a Gabor patch.
    - self.generate_sequences()
        Generates a set of sequences.
    - self.get_all_oris(image_mean_oris)
        Gets exact orientations per patch from mean orientations. 
    - self.get_image_mean_oris(seq_images)
        Gets mean orientations for a sequence of images.
    - self.get_sequence_idxs()
        Gets indices to index into full continuous sequence into order to obtain 
        individual sequences.
    - self.get_sequence_image_types()
        Gets a sequence of image types.
    - self.image_class_to_label(seq_image_types, seq_image_mean_oris)
        Converts class names to labels.
    - self.image_label_to_class(seq_labels)
        Converts class labels to names.
    - self.image_to_image_idx(seq_image_types)
        Obtains indices for positions and sizes arrays for image types. 
    - self.overlay_gabors(image_gabor_patches, positions)
        Overlays gabor patches at specific positions.
    - self.sample_mean_ori()
        Samples a mean orientation value.
    - self.set_deg_per_pix(deg_width)
        Sets the visual degree to pixel conversion to use.
    - self.set_num_cycles_adj(num_cycles)
        Sets the adjusted number cycles to use in drawing Gabor patches.  
    - self.set_possizes()
        Sets the positions and sizes for the dataset, based on an input seed.
    - self.set_size_ran(size_ran)
        Sets the size range for the Gabors (in degrees).
    - self.set_transform(transform)
        Sets the torch Transform used for the dataset.
    -------
    """

    def __init__(
        self, 
        width=256, 
        height=256, 
        mode="train",
        dataset_name="Gabors",
        transform=None,
        train_len=5000,
        seq_len=10,
        num_seq=5,
        num_gabors=NUM_GABORS, 
        num_mean_oris=gabor_utils.NUM_MEAN_ORIS,
        gab_img_len=GAB_IMG_LEN,
        same_possizes=True,
        gray=True, 
        roll=False, 
        shift_frames=False,
        unexp=True,
        U_prob=0.1, 
        diff_U_possizes=False, 
        size_ran=[10, 20],
        deg_width=120,
        ori_std_dev=0.25,
        num_cycles=1,
        psi=0.25,
        gamma=1,
        return_label=False,
        supervised=False,
        seed=None,
        ):
        """
        GaborSequenceGenerator()

        Constructs a GaborSequenceGenerator object

        Optional args
        -------------
        - width : int (default=256)
            Individual Gabor image width (pixels)
        - height : int (default=256)
            Individual Gabor image height (pixels)
        - mode : str (default="train")
            Dataset mode (i.e., "train", "val", or "test") 
        - dataset_name : str (default="Gabors")
            Dataset name
        - transform : torch Transform (default=None)
            Transform to apply to the Gabor sequences sampled.
        - train_len : int (default=5000)
            Dataset length, in train mode.
        - seq_len : int (default=10)
            Number of frames per sequence.
        - num_seq : int (default=5)
            Number of consecutive sequences to return per sample.
        - num_gabors : int(default=NUM_GABORS)
            Number of Gabor patches per image
        - num_mean_oris : int(default=gabor_utils.NUM_MEAN_ORIS)
            Number of equally spaced orientations to sampled from, 
            between 0 and 360 
            degrees. 
        - gab_img_len : int(default=GAB_IMG_LEN)
            Number of frames to repeat for Gabor image
        - same_possizes : bool (default=True)
            If True, positions and sizes are fixed for the dataset.
        - gray : bool (default=True)
            If True, grayscreen images are included in the base sequence.
        - roll : bool (default=False)
            If True, the base sequence can be rolled when generating the full 
            continuous sequence.
        - shift_frames : bool (default=False)
            If True and roll is True, rolling also allows shifting the start 
            point of a sequence down to the level of repeated frames.
        - unexp : bool (default=True)
            If True, dataset is set to produce unexpected sequences
        - U_prob : float (default=0.1)
            Probability of replacing each individual D with a U frame, if 
            unexp is True. 
        - diff_U_possizes : bool (default=False)
            Whether U positions and sizes are different from D positions and 
            sizes.
        - size_ran : list (default=[10, 20])
            Size range for the Gabor patches (in visual degrees).
        - deg_width : float (default=120)
            Image width in visual degrees, if it were curved (used for 
            conversion to pixels).
        - ori_std_dev : float (default=0.25)
            Orientation standard deviation (in radians).
        - num_cycles : float (default=1)
            Approximate number Gabor cycles that should be visible for each 
            Gabor patch (only implemented between 0.5 and 3).
        - psi : float (default=0.25)
            Gabor patch phase (as a proportion of a full cycle).
        - gamma : float (default=1)
            Gabor patch spatial aspect ratio (width to height ratio).
        - return_label : bool (default=False)
            If True, when sequences are sampled, labels for each image are 
            returned as well.
        - supervised : bool (default=False)
            If True, dataset is set to supervised mode.
        - seed : int (default=None)
            Seed to use to set positions and sizes, if same_possizes is True.
        """
        
        self.mode         = mode
        self.dataset_name = dataset_name
        self.supervised   = supervised

        self.width           = int(width)
        self.height          = int(height)
        self.seq_len         = int(seq_len)
        self.num_seq         = int(num_seq)
        self.num_gabors      = int(num_gabors)
        self.num_mean_oris   = int(num_mean_oris)
        self.gab_img_len     = int(gab_img_len)

        if self.mode == "train":
            self.dataset_len = int(train_len)
        else:
            self.dataset_len = int(train_len / 2)

        self.full_len        = int(seq_len * num_seq)
        if self.mode == "test" and self.supervised:
            self.full_len *= 5

        self.same_possizes   = bool(same_possizes)
        self.gray            = bool(gray)
        self.roll            = bool(roll)
        self.shift_frames    = bool(shift_frames)
        self.unexp           = bool(unexp)
        self.diff_U_possizes = bool(diff_U_possizes)
        
        self.kappa              = 1.0 / (ori_std_dev ** 2)
        self.psi_rad            = 2 * psi * np.pi
        self.gamma              = gamma
        self.num_sigmas_visible = 3
        
        # initialize relevant properties
        self._set_U_prob(U_prob)
        self.set_num_cycles_adj(num_cycles)
        self.set_deg_per_pix(deg_width)
        self.set_size_ran(size_ran)
        self.set_transform(transform)
        self.num_base_seq_repeats
        self.mean_oris
        self.image_type_dict

        if self.same_possizes:
            self.set_possizes(seed=seed)

        if self.supervised:
            return_label = True
        self.return_label = return_label


    def _set_U_prob(self, U_prob=0.1):
        """
        self._set_U_prob()

        Sets the probability of replacing individual D images with U images.

        Optional args
        -------------
        - U_prob : float (default=0.1)
            Probability of replacing an individual D image with a U image.
        """

        if hasattr(self, "U_prob"):
            raise AttributeError(
                "Cannot reset this attribute after initialization, as other "
                "attributes are tied to it."
                )
        
        if U_prob < 0 or U_prob > 1:
            raise ValueError("U_prob must be between 0 and 1, inclusively.")
        self.U_prob = U_prob


    def set_num_cycles_adj(self, num_cycles=1):
        """
        self.set_num_cycles_adj()

        Sets the probability of replacing individual D images with U images.

        Optional args
        -------------
        - num_cycles : float (default=1)
            Approximate number Gabor cycles that should be visible for each 
            Gabor patch (only implemented between 0.5 and 3).
        """

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

        Sets the visual degrees to pixel conversion to use in generating Gabor 
        patches.

        Optional args
        -------------
        - deg_width : float (default=120)
            Image width in visual degrees.
        - curved : bool (default=True)
            If True, deg_width is the image width in visual degrees, if the 
            image were curved (equidistant in all points from the eye). 
            Otherwise, if it were flat.
        """
        
        if curved:
            self.deg_per_pix = deg_width / self.width
        else:
            dist = self.width / (2 * np.tan(np.deg2rad(deg_width / 2)))
            self.deg_per_pix = np.rad2deg(np.arctan(1 / dist))


    def set_size_ran(self, size_ran=[10, 20]):
        """
        self.set_size_ran()

        Sets the size range (in visual degrees) from which to sample Gabor 
        patch sizes.

        Optional args
        -------------
        - size_ran : list (default=[10, 20])
            Size range in visual degrees [min, max].
        """

        if len(size_ran) != 2:
            raise ValueError("'size_ran' should have length 2.")
        if size_ran[0] <= 0 or size_ran[1] <= 0:
            raise ValueError("'size_ran' values must be strictly positive.")
        if size_ran[0] > size_ran[1]:
            size_ran = [size_ran[1], size_ran[0]]

        self.size_ran = size_ran


    def set_transform(self, transform=None):
        """
        set_transform()

        Sets the Tensor to use on Gabor sequences, once generated.

        Optional args
        -------------
        - transform : torch Transform (default=None)
            Transform to use. If None, a minimal transform is set to convert 
            images to tensor and normalize them.
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

        - 1D array
            Mean orientations to sample from for expected image types.
        """
        
        if not hasattr(self, "_mean_oris"):
            self._mean_oris = gabor_utils.get_mean_oris(self.num_mean_oris)
        return self._mean_oris


    @property
    def mean_oris_U(self):
        """
        self.mean_oris_U

        - 1D array
            Mean orientations to sample from for unexpected image types 
            (i.e., "U").
        """
        
        if not hasattr(self, "_mean_oris_U"):
            self._mean_oris_U = gabor_utils.get_mean_U_oris(self.mean_oris) 
        return self._mean_oris_U


    @property
    def all_mean_oris(self):
        """
        self.all_mean_oris

        - 1D array
            All possible mean orientations (for expected or unexpected image 
            types).
        """
        
        if not hasattr(self, "_all_mean_oris"):
            if self.U_prob > 0:
                mean_oris = np.concatenate([self.mean_oris, self.mean_oris_U])
            else:
                mean_oris = self.mean_oris
            self._all_mean_oris = np.sort(np.unique(mean_oris))
        return self._all_mean_oris


    @property
    def base_seq(self):
        """
        self.base_seq

        - list
            Base image type sequence (e.g., ["A", "B", "C", "D", "G"]).
        """
        
        if not hasattr(self, "_base_seq"):
            self._base_seq = copy.deepcopy(gabor_utils.BASE_SEQ)
            if self.gray:
                self._base_seq.append("G")
        return self._base_seq


    @property
    def num_frames_total(self):
        """
        self.num_frames_total

        - int
            Total number of frames per sequence.
        """
        
        if not hasattr(self, "_num_frames_total"):
            self._num_frames_total = int(
                np.ceil(self.full_len / self.gab_img_len)
                ) + int(self.roll) # add a buffer of 1, if rolling
        return self._num_frames_total


    @property
    def num_base_seq_repeats(self):
        """
        self.num_base_seq_repeats

        - int
            Number of base sequence repeats (minimum) needed to generate a full 
           continuous sequences.
        """

        if not hasattr(self, "_num_base_seq_repeats"):
            self._num_base_seq_repeats = int(
                np.ceil(self.num_frames_total / len(self.base_seq))
                )
        return self._num_base_seq_repeats


    @property
    def image_type_dict(self):
        """
        self.image_type_dict

        - dict
            Dictionary for associating image types (e.g., "A", "B", etc.) to 
            indices for the positions and sizes arrays.

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
    def class_dict_encode(self):
        """
        self.class_dict_encode

        - dict
            Dictionary for decoding class names from class labels.
        """
        
        if not hasattr(self, "_class_dict_encode"):
            self._class_dict_encode = \
                gabor_utils.get_image_class_and_label_dict(
                    num_mean_oris=self.num_mean_oris, 
                    gray=self.gray, 
                    U_prob=self.U_prob, 
                    diff_U_possizes=self.diff_U_possizes,
                    class_to_label=True
                    )

        return self._class_dict_encode


    @property
    def class_dict_decode(self):
        """
        self.class_dict_decode

        - dict
            Dictionary for encoding class names into class labels.
        """

        if not hasattr(self, "_class_dict_decode"):
            self._class_dict_decode = {
                label: im_cl for im_cl, label in self.class_dict_encode.items()
                }

        return self._class_dict_decode
        

    @property
    def positions(self):
        """
        - 3D array
            Gabor patch positions for each image type, with 
            dims: image type x [height, width] x number of gabors
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
        - 2D array
            Gabor patch sizes for each image type (in visual degrees), with 
            dims: image type x number of gabors
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

        Gets indices to index into full continuous sequence into order to 
        obtain individual sequences.

        Optional args
        -------------
        - get_num_seq : bool (default=True)
            If True, indices for self.num_seq consecutive sequences of length 
            self.seq_len are returned. Otherwise, indices for all possible 
            sequences of length self.seq_len are returned. In all cases, 
            sequences are non-overlapping.

        Returns
        -------
        - sequence_idxs : 2D Tensor
            Sequence indices, with dims: number of sequences x seq length
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
        
        sequence_idxs = torch.from_numpy(seq_idx)

        return sequence_idxs


    @property
    def sub_batch_idxs(self):
        """
        self.sub_batch_idxs

        - 2D Tensor
            Indices for indexing into a full continous sequence to generate 
            overlapping sub-batches (used in "test" mode).
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

        Sets, or optionally resets, the sizes and positions attributes, based 
        on a seed.

        Optional args
        -------------
        - seed : int (default=None)
            Seed to use to initialize the random state object used to set the 
            positions and sizes.
        - reset : bool (default=True)
            If True, existing sizes and positions attributes are replaced. 
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

        Samples a mean orientation (for expected image types).

        Returns
        -------
        - mean_ori : float
            Sampled mean orientation (in degrees).
        """

        mean_ori = np.random.choice(self.mean_oris)
        
        return mean_ori


    def image_to_image_idx(self, seq_image_types):
        """
        self.image_to_image_idx(seq_image_types)

        Returns image type indices for indexing into the positions and sizes 
        arrays for each image type.

        Required args
        -------------
        - seq_image_types : 1D array-like
            Sequence of image types.

        Returns
        -------
        - seq_image_idxs : 1D array
            Sequence of image type indices. 
        """

        seq_image_idxs = np.asarray(
            [self.image_type_dict[i] for i in seq_image_types]
            )

        return seq_image_idxs


    def image_class_to_label(self, seq_image_types, seq_image_mean_oris):
        """
        self.image_class_to_label(seq_image_types, seq_image_mean_oris)

        Converts class names to labels.
        
        Required args
        -------------
        - seq_image_types : array-like
            Image types (e.g., "A", "B", etc.)
        - seq_image_mean_oris : array-like
            Gabor orientations (same shape as seq_image_types, and numerical 
            only)

        Returns
        -------
        - seq_image_labels : nd Tensor
            Class labels for the input image types and orientations (in degrees)
            (same shape as seq_image_types).
        """

        seq_image_labels = gabor_utils.image_class_to_label(
            self.class_dict_encode, 
            seq_image_types, 
            seq_image_mean_oris, 
            class_to_label=True
            )

        return seq_image_labels


    def image_label_to_class(self, seq_labels, seq_unexp=None):
        """
        self.image_label_to_class(seq_labels)

        Converts class labels to names.
        
        Required args
        -------------
        - seq_labels : array-like
            Image class labels

        Optional args
        -------------
        - seq_unexp : array-like (default=None)
            Boolean array indicating which image types are unexpected ("U")
            (same shape as seq_labels). If not provided, any "D/U" values are 
            retained.

        Returns
        -------
        - seq_classes : list
            Class names for the input class labels (same structure as 
            seq_labels), where each class is (image_type, orientation 
            (in degrees)).
        """

        seq_classes = gabor_utils.image_label_to_class(
            self.class_dict_decode, 
            seq_labels, 
            label_to_class=True,
            seq_unexp=seq_unexp
            )

        return seq_classes


    def _replace_Ds_with_Us(self, seq_image_types):
        """
        self._replace_Ds_with_Us(seq_image_types)

        Replaces D image types with Us probabilistically.
        
        Required args
        -------------
        - seq_image_types : 1D array-like
            Image types (e.g., "A", "B", etc.)

        Returns
        -------
        - seq_image_types : 1D array-like
            Image types (e.g., "A", "B", etc.)
        """
        
        seq_image_types = seq_image_types[:]
        if not self.unexp or self.U_prob == 0:
            return seq_image_types
        
        num_Ds  = seq_image_types.count("D")
        if num_Ds == 0:
            return seq_image_types

        Us = np.random.rand(num_Ds) < self.U_prob 
        if Us.sum():
            U_idxs = np.where(np.asarray(seq_image_types) == "D")[0][Us]
            for U_idx in U_idxs:
                seq_image_types[U_idx] = "U"

        return seq_image_types


    def get_sequence_image_types(self):
        """
        self.get_sequence_image_types()

        Gets a sequence of image types, based on repetitions of the base 
        sequence.

        Returns
        -------
        - seq_image_types : list
            Sequence of image types (e.g., "A", "B", etc.)
        """
        
        if self.roll:
            roll_shift = np.random.choice(len(self.base_seq))
            seq_image_types = np.roll(self.base_seq, roll_shift).tolist()
        else:
            seq_image_types = self.base_seq[:] # make a copy
        
        seq_image_types = seq_image_types * self.num_base_seq_repeats
        seq_image_types = seq_image_types[: self.num_frames_total]

        seq_image_types = self._replace_Ds_with_Us(seq_image_types)

        return seq_image_types


    def get_image_mean_oris(self, seq_image_types):
        """
        self.get_image_mean_oris(seq_image_types)

        Gets mean orientations based on a sequence of image types.
        
        Required args
        -------------
        - seq_image_types : 1D array-like
            Sequence of image types (e.g., "A", "B", etc.)

        Returns
        -------
        - image_mean_oris : 1D array
            Mean orientations for each image 
            (0-360 degrees, and NaNs for "G" images).        
        """

        image_idxs = self.image_to_image_idx(seq_image_types)
        min_image_idx = min(image_idxs)
        ori_reset_idxs = np.where(np.asarray(image_idxs) == min_image_idx)[0]
        
        image_mean_oris = []
        mean_ori = self.sample_mean_ori()
        for i, seq_image_type in enumerate(seq_image_types):
            if i in ori_reset_idxs:
                mean_ori = self.sample_mean_ori()
            ori = mean_ori
            if seq_image_type == "U":
                ori = (mean_ori + gabor_utils.U_ADJ) % 360
            elif seq_image_type == "G":
                ori = np.nan
            image_mean_oris.append(ori)

        image_mean_oris = np.asarray(image_mean_oris)
        
        return image_mean_oris


    def get_all_oris(self, image_mean_oris):
        """
        self.get_all_oris(image_mean_oris)

        Returns specific orientations for each Gabor patch in a sequence.
        
        Required args
        -------------
        - image_mean_oris : 1D array
            Mean orientations for each image 
            (0-360 degrees, and NaNs for "G" images).

        Returns
        -------
        - all_oris : 2D array
            Specific orientations (0-360 degrees, and NaNs for "G" images), 
            with dims: number of images x number of Gabor patches 
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

        Creates a Gabor patch.

        Required args
        -------------
        - gabor_size : int
            Gabor size (in pixels).

        Optional args
        -------------
        - theta : float (default=0)
            Gabor orientation (in radians).

        Returns
        -------
        - gabor_filter : 2D array
            Gabor filter patch, with dims: height x width
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

        Overlays Gabor patches at the specified positions.

        Required args
        -------------
        - image_gabor_patches : list
            Gabor patches, each with dims [patch height, patch width].
        - positions : 3D array
            Gabor patch positions, with dims: [height, width] x number of gabors

        Returns
        -------
        - image : 2D array
            Overlayed Gabor patches, normalized to uint8, and 
            with dims: height x width
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

        Returns a single full continuous sequence, and extended labels.
    
        The extended labels include the class label for each image, and a 
        boolean value indicating whether the image is of the unexpected U type.

        Returns
        -------
        - gabor_seq : 4D array
            Gabor image sequence, 
            with dims: number of images x height x width x color channels (3)
        - seq_ext_labels : 2D array
            with dims: number of images x labels [class label, unexp]
        """

        # identify images and orientations for the full sequence        
        seq_image_types = self.get_sequence_image_types()
        seq_image_mean_oris = self.get_image_mean_oris(seq_image_types)

        # Get orientation per image and patch
        all_seq_oris = self.get_all_oris(seq_image_mean_oris)
        all_seq_oris = np.deg2rad(all_seq_oris) # convert to radians

        # Get sizes
        seq_image_type_idxs = self.image_to_image_idx(seq_image_types)
        sizes = self.sizes[seq_image_type_idxs]
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
        positions = self.positions[seq_image_type_idxs]
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
        seq_labels = self.image_class_to_label(
            seq_image_types, seq_image_mean_oris
            ).numpy()
        seq_unexp_labels = gabor_utils.get_unexp(seq_image_types).numpy()
        seq_ext_labels = np.vstack([seq_labels, seq_unexp_labels]).T
        seq_ext_labels = np.repeat(seq_ext_labels, self.gab_img_len, axis=0)

        # Randomly shift the start point
        if self.roll and self.shift_frames and self.gab_img_len > 1:
            shift = np.random.choice(self.gab_img_len)
            gabor_seq = gabor_seq[shift : shift + self.full_len]
            seq_ext_labels = seq_ext_labels[shift : shift + self.full_len]
        
        return gabor_seq, seq_ext_labels


    def generate_sequences(self):
        """
        self.generate_sequences()

        Returns sequences, and their extended labels. 
        
        The extended labels include the class label for each image, and a 
        boolean value indicating whether the image is of the unexpected U type.

        Returns
        -------
        - gabor_seq : 5D Tensor
            Gabor image sequences, 
            with dims: 
                number of seq x color channels (3) x seq len x height x width
        - seq_ext_labels : 3D Tensor
            with dims: 
                number of seq x number of images x labels [class label, unexp]
        """

        gabor_seq, seq_ext_labels = self._generate_sequence()

        gabor_seq = [
            Image.fromarray(gabor_img, mode="RGB") for gabor_img in gabor_seq
            ]
        
        gabor_seq = self.transform(gabor_seq) # apply transform

        gabor_seq = torch.stack(gabor_seq, 0)

         # Convert to torch 
        seq_ext_labels = torch.from_numpy(seq_ext_labels)

        # Cut into sequences
        if self.mode == "test" and self.supervised:
            get_num_seq = False
        else:
            get_num_seq = True

        seq_idx = self.get_sequence_idxs(get_num_seq=get_num_seq)
        gabor_seq = gabor_seq[seq_idx] # N_all x SL x C x H x W
        seq_ext_labels = seq_ext_labels[seq_idx]

        # Move color dimension
        gabor_seq = torch.moveaxis(gabor_seq, 2, 1)

        if self.mode == "test" and self.supervised:
            # SUB_B x N x C x SL x H x W
            gabor_seq = gabor_seq[self.sub_batch_idxs]
            seq_ext_labels = seq_ext_labels[self.sub_batch_idxs]

        return gabor_seq, seq_ext_labels


    def __getitem__(self, ix):
        """
        Returns sequences, and, if self.return_label is True, their extended 
        labels. 
        
        The extended labels include the class label for each image, and a 
        boolean value indicating whether the image is of the unexpected U type.

        Required args
        -------------
        - ix : int
            Sampling index (ignored)

        Returns
        -------
        - gabor_seq : 5D Tensor
            Gabor image sequences, 
            with dims: 
                number of seq x color channels (3) x seq len x height x width
        
        if self.return_label:
        - seq_ext_labels : 3D Tensor
            with dims: 
                number of seq x number of images x labels [class label, unexp]
        """

        gabor_seq, seq_ext_labels = self.generate_sequences()
        
        if self.return_label:
            return gabor_seq, seq_ext_labels
        else:
            return gabor_seq

    
    def __len__(self):
        """
        Returns a preset dataset length value.
        """
        
        return self.dataset_len
        
