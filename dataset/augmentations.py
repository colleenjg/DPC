import random
import numbers
import math
import collections
import numpy as np
from PIL import ImageOps, Image

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F


#############################################
class Padding(object):
    """
    Augmentation for padding an image with 0 values.

    Attributes
    ----------
    - pad : int or tuple
        Padding to apply to image: either a single value for all sides, 
        2 values (1 per dimension) or 4 values (1 per side).

    Methods
    -------
    - self.__call__(img):
        Applies padding to an image.
    """

    def __init__(self, pad):
        """
        Padding(pad)

        Constructs a Padding object.

        Required args
        -------------
        - pad : int or tuple
            Padding to apply to image: either a single value for all sides, 
            2 values (1 per dimension) or 4 values (1 per side).
        """

        self.pad = pad

    def __call__(self, img):
        """
        self.__call__(img)

        Applies padding.

        Required args
        -------------
        - img: PIL Image
            PIL Image

        Returns
        -------
        - modified PIL Image
            Padded PIL Image.
        """

        return ImageOps.expand(img, border=self.pad, fill=0)


#############################################
class Scale(object):
    """
    Augmentation for scale a list of PIL images to a specified size.

    Attributes
    ----------
    - size : int or iterable
        Size to scale to: either a single value for both dimensions, an 
        iterable of length 2.
    - interpolation : int
        Index of the PIL resampling filter to use (see Image.resize() method).

    Methods
    -------
    - self.__call__(imgmap):
        Applies cropping to images.
    """

    def __init__(self, size, interpolation=Image.NEAREST):
        """
        Scale(size)

        Constructs a Scale object.

        Required args
        -------------
        - size : int or iterable
            Size to scale to: either a single value for both dimensions, an 
            iterable of length 2.

        Optional args
        -------------
        - interpolation : int (default=Image.NEAREST)
            Index of the PIL resampling filter to use (see Image.resize()
            method).
        """

        assert (
            isinstance(size, int) or 
            (isinstance(size, collections.Iterable) and len(size) == 2)
        )
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies scaling.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images, all with the same dimensions

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, each scaled.
        """

        # assert len(imgmap) > 1 # list of images
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


#############################################
class CenterCrop(object):
    """
    Augmentation for center cropping a list of PIL images to a specified size.

    Attributes
    ----------
    - size : int or iterable or None
        Size to crop to: either a single value for both dimensions, an 
        iterable of length 2, or None for no cropping.

    Methods
    -------
    - self.__call__(imgmap):
        Applies cropping to images.
    """
    
    def __init__(self, size, consistent=True):
        """
        CenterCrop(size)

        Constructs a CenterCrop object.

        Required args
        -------------
        - size : int or iterable or None
            Size to crop to: either a single value for both dimensions, an 
            iterable of length 2, or None for no cropping.

        Optional args
        -------------
        - consistent : bool (default=True)
            Ignored.
        """

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies center cropping.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images, all with the same dimensions

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, each center cropped
        """

        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


#############################################
class RandomCropWithProb(object):
    """
    Augmentation for randomly cropping a list of PIL images to a specified 
    size, given a specified probability.

    Attributes
    ----------
    - size : int or iterable or None
        Size to crop to: either a single value for both dimensions, an 
        iterable of length 2, or None for no cropping.
    - consistent : bool
        Whether to crop consistently across images.
    - threshold : float
        Probability threshold for randomly cropping images size.

    Methods
    -------
    - self.__call__(imgmap):
        Applies cropping to images.
    """

    def __init__(self, size, p=0.8, consistent=True):
        """
        RandomCropWithProb(size)

        Constructs a RandomCropWithProb object.

        Required args
        -------------
        - size : int or iterable or None
            Size to crop to: either a single value for both dimensions, an 
            iterable of length 2, or None for no cropping.

        Optional args
        -------------
        - threshold : float (default=0.8)
            Probability threshold for cropping images to the specified size.
        - consistent : bool (default=True)
            If True, all images passed together are cropped using the same 
            coordinates, instead of new cropping coordinates being sampled for 
            each image.
        """

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies cropping, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images, all with the same dimensions

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, each cropped
        """
        
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w - tw) / 2.))
                    y1 = int(round((h - th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.))
                        y1 = int(round((h - th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


#############################################
class RandomCrop(object):
    """
    Augmentation for randomly cropping a list of PIL images to a specified 
    size.

    Attributes
    ----------
    - size : int or iterable or None
        Size to crop to: either a single value for both dimensions, an 
        iterable of length 2, or None for no cropping.
    - consistent : bool
        Whether to crop consistently across images.

    Methods
    -------
    - self.__call__(imgmap):
        Applies cropping to images.
    """

    def __init__(self, size, consistent=True):
        """
        RandomCrop(size)

        Constructs a RandomCrop object.

        Required args
        -------------
        - size : int or iterable or None
            Size to crop to: either a single value for both dimensions, an 
            iterable of length 2, or None for no cropping.

        Optional args
        -------------
        - consistent : bool (default=True)
            If True, all images passed together are cropped using the same 
            coordinates, instead of new cropping coordinates being sampled for 
            each image.
        """

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        """
        self.__call__(imgmap)

        Applies cropping and rescaling, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images, all with the same dimensions

        Optional args
        -------------
        - flowmap: 4D+ array (default=None)
            Flow map array, of the same size as the Array or list of PIL 
            Images, used to guide cropping.
            If provided, self.consistent must be False. Each image will be 
            cropped, to an area that maximizes optical flow, given 3 randomly 
            selected options. 
            Dimensions: num_images x height x width x color

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, each cropped
        """

        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            else:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    # create proposals and use the one with largest optical flow
                    for j in range(3): 
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([
                            x, y, abs(np.mean(flowmap[idx, y:y+th, x:x+tw, :]))
                            ])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


#############################################
class RandomSizedCrop(object):
    """
    Augmentation for either center cropping a list of PIL images to a specified 
    size, or cropping them to a random size and rescaling them, given a 
    specified probability.

    Attributes
    ----------
    - size : iterable
        Size to crop to (length 2).
    - interpolation : int
        Index of the PIL resampling filter to use (see Image.resize() method).
    - consistent : bool
        Whether to crop consistently across images.
    - threshold : float
        Probability threshold for randomly cropping and rescaling, instead of 
        center cropping to the specified size.
    
    Methods
    -------
    - self.__call__(imgmap):
        Applies cropping and scaling to images.
    """

    def __init__(self, size, interpolation=Image.BILINEAR, consistent=True, 
                 p=1.0):
        """
        RandomSizedCrop(size)

        Constructs a RandomSizedCrop object.

        Required args
        -------------
        - size : int
            Size to crop both dimensions to.

        Optional args
        -------------
        - interpolation : int (default=Image.BILINEAR)
            Index of the PIL resampling filter to use (see Image.resize() 
            method).
        - consistent : bool (default=True)
            If True, all images passed together are cropped using the same 
            coordinates, instead of new cropping coordinates being sampled for 
            each image. Note that in all cases, the cropping size, and 
            rescaling is consistent across images.
        - p : float (default=1.0)
            Probability of randomly cropping and rescaling, instead of center 
            cropping to the specified size.
        """

        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p 

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies cropping and rescaling, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images, all with the same dimensions

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, each cropped and scaled
        """
        
        img1 = imgmap[0]
        if random.random() < self.threshold: # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(0.5, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [
                            i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap
                            ]
                        for i in imgmap: assert(i.size == (w, h))

                        return [
                            i.resize((self.size, self.size), self.interpolation) 
                            for i in imgmap
                            ]
                else:
                    result = []
                    for i in imgmap:
                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = random.randint(0, img1.size[0] - w)
                            y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert(result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [
                        i.resize((self.size, self.size), self.interpolation) 
                        for i in result
                        ] 

            # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else: # don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size)
            return crop(imgmap)


#############################################
class RandomHorizontalFlip(object):
    """
    Augmentation for flipping images horizontally, with a specific probability.

    Attributes
    ----------
    - consistent : bool
        Whether to flip consistently across images.
    - threshold : float
        Probability threshold for flipping images horizontally.

    Methods
    -------
    - self.__call__(imgmap):
        Applies flipping to images.
    """

    def __init__(self, consistent=True, command=None):
        """
        RandomHorizontalFlip()

        Constructs a RandomHorizontalFlip object.

        Optional args
        -------------
        - consistent : bool (default=True)
            If True, all images passed together are flipped consistently, 
            instead of individually.
        - command : str or None (default=None)
            Controls the flipping probability threshold
            "left": sets threshold to 0 (never flip)
            "right": sets threshold to 1 (always flip)
            otherwise: sets threshold to 0.5
        """

        self.consistent = consistent
        if command == "left":
            self.threshold = 0
        elif command == "right":
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies flipping, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images, all with the same dimensions.

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, optionally flipped.
        """

        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i) 
            assert len(result) == len(imgmap)
            return result 


#############################################
class RandomGray(object):
    """
    Augmentation for converting images to grayscale based on a randomly 
    selected channel.

    Note: This augmentation actually selects a single channel for each image, 
    instead of fully converting all channels to grayscale.

    Attributes
    ----------
    - consistent : bool
        Whether to convert all images to grayscale, instead of each 
        individually (see __init__ for more details)
    - threshold : float
        Probability threshold for randomly cropping and rescaling, instead of 
        center cropping to the specified size.
    
    Methods
    -------
    - self.__call__(imgmap):
        Applies cropping and scaling to images.

    """
    def __init__(self, consistent=True, p=0.5):
        """
        - consistent : bool (default=True)
            Whether to convert all images to grayscale, instead of each 
            individually.
            Note that the channel selection itself is not consistent, even if 
            consistent is True.

        - p : float (default=0.5)
            Probability of randomly converting images to grayscale.
        """
        
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies grayscale conversion, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images.

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, optionally converted to grayscale.
        """
        
        if self.consistent:
            if random.random() < self.threshold:
                return [self.grayscale(i) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(self.grayscale(i))
                else:
                    result.append(i) 
            assert len(result) == len(imgmap)
            return result 

    def grayscale(self, img):
        """
        self.grayscale(img)

        Randomly selects a channel to copy to all channels.

        Required args
        -------------
        - img: PIL Image
            3 channel PIL Image.

        Returns
        -------
        - img: PIL Image
            Grayscale PIL Image in RGB format, created from one channel.
        """
        
        channel = np.random.choice(3)
        np_img = np.array(img)[:, :, channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, "RGB")
        return img 


#############################################
class ColorJitter(object):
    """ 
    Augmentation for jittering color (brightness, contrast, saturation and hue) 
    for a list of Tensor images, with a specified probability.

    Attributes
    ----------
    - brightness : float or tuple of floats (min, max)
        Brightness range (see __init__ for more details)
    - contrast : float or tuple of floats (min, max)
        Contrast range (see __init__ for more details)
    - saturation : float or tuple of floats (min, max)
        Saturation range (see __init__ for more details)
    - hue : float or tuple of floats (min, max)
        Hue range (see __init__ for more details)
    - consistent : bool
        Whether to jitter color consistently across images.
    - threshold
        Probability threshold for applying the color jitter changes to images.
    
    Methods
    -------
    - self._check_input(value, name)
        Checks whether input values are acceptable and returns the inferred 
        final values.

    - self.__call__(imgmap):
        Applies color jitter to images.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, 
                 consistent=False, p=1.0):
        """
        ColorJitter

        Constructs a ColorJitter object.

        Optional args
        -------------
        - brightness : float or tuple of floats (min, max) (default=0) 
            How much to jitter brightness, where the brightness factor is 
            sampled uniformly from [max(0, 1 - brightness), 1 + brightness] or 
            the given [min, max] values. 
            Should be non negative numbers.
        - contrast : float or tuple of floats (min, max) (default=0)
            How much to jitter contrast (see brightness for details on how 
            contrast factors are set when sampled).
        - saturation : float or tuple of floats (min, max) (default=0)
            How much to jitter saturation (see brightness for details on how 
            contrast factors are set when sampled).
        - hue : float or tuple of floats (min, max) (default=0)
            How much to jitter hue, where the hue factor is sampled uniformly 
            from [-hue, hue] or the given [min, max] values. 
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        - consistent : bool (default=False)
            If True, all images passed together are color jittered 
            consistently, instead of individually.
        - p : float (default=1.0)
            Probability of randomly jittering the color of all of the images.
        """

        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p 

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), 
                     clip_first_on_zero=True):
        """
        self._check_input(value, name)

        Converts value to a range or checks values against bounds if it is 
        already a range.  

        Required args
        -------------
        - value : float or tuple of floats (min, max)
            Value from which to determine the range for the parameter: either
            [max(0, 1 - value), 1 + value] if value is a number, or [min, max] 
            if value is a list or tuple. 
        - name : str
            The name of the parameter for which value is being passed.

        Optional args
        -------------
        - center : float (default=1)
            Center of the range to construct, if a single value is passed.
        - bound : tuple (default=(0, float("inf"))
            Min and max bounds (inclusive) if value is a tuple or list. 
        - clip_first_on_zero : bool (default=True)
            If True, and a single value is passed, the minimum value is clipped 
            to 0.

        Returns
        -------
        - value : tuple, list or None
            range from which to sample the value [min, max] or 
            None if the range corresponds exactly to the center
        """
        
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                    )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with "
                "length 2."
                )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness=None, contrast=None, saturation=None, hue=None):
        """
        get_params()
        
        Gets a randomized color jitter transform to be applied to an image.

        Optional args
        -------------
        - brightness : tuple of floats (min, max) (default=None) 
            Range from which to sample brightness factor. 
            Should be non negative numbers.
        - contrast : tuple of floats (min, max) (default=None)
            Range from which to sample contrast factor. 
            Should be non negative numbers.
        - saturation : tuple of floats (min, max) (default=None)
            Range from which to sample saturation factor. 
            Should be non negative numbers.
        - hue : tuple of floats (min, max) (default=None)
            Range from which to sample hue factor. 
            Should be between -0.5 and 0.5.

        Returns
        -------
        - transform : torch transform
            Torch transform for adjusting brightness, contrast and saturation, 
            in a random order.
        """

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_brightness(img, brightness_factor)
                    )
                )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_contrast(img, contrast_factor)
                    )
                )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_saturation(img, saturation_factor)
                    )
                )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_hue(img, hue_factor)
                    )
                )

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies color jitter, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images.

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, optionally color jittered.
        """
        
        if random.random() < self.threshold: # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                result = []
                for img in imgmap:
                    transform = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
                    result.append(transform(img))
                return result
        else: # don't do ColorJitter, do nothing
            return imgmap 

    def __repr__(self):
        """
        self.__repr__()

        Returns string listing the color jitter ranges for each parameter.
        """
        
        format_string = (
            f"{self.__class__.__name__} ("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"hue={self.hue})"
        )
        return format_string


#############################################
class RandomRotation(object):
    """
    Augmentation for rotating a list of Tensor images, with a specified 
    probability.

    Attributes
    ----------
    - consistent : bool
        Whether to apply rotation consistently across images.
    - degree : bool
        Maximum rotation (+/-) to apply.
    - threshold : float
        Probability threshold for rotating images.
    
    Methods
    -------
    - self.__call__(imgmap):
        Applies rotation to images.
    """

    def __init__(self, consistent=True, degree=15, p=1.0):
        """
        RandomRotation()

        Constructs a RandomRotation object.

        Optional args
        -------------
        - consistent : bool (default=True)
            If True, all images passed together are rotated consistently, 
            instead of individually.
        - degree : int (default=15)
            Maximum rotation (+/-) to apply.
        - p : float (default=1.0)
            Probability of rotating images.
        """
        
        self.consistent = consistent
        self.degree = degree 
        self.threshold = p

    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies rotations, if applicable.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images.

        Returns
        -------
        - list of modified PIL Images
            List of PIL Images, optionally rotated.
        """
        
        if random.random() < self.threshold: # do RandomRotation
            if self.consistent:
                deg = np.random.randint(-self.degree, self.degree, 1)[0]
                return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                return [
                    i.rotate(
                        np.random.randint(-self.degree, self.degree, 1)[0], 
                        expand=True
                        ) 
                    for i in imgmap
                ]
        else: # skip RandomRotation
            return imgmap 


#############################################
class ToTensor(object):
    """
    Augmentation for converting a list of PIL images to Tensor images.

    Methods
    -------
    - self.__call__(imgmap):
        Converts images to Tensor images.
    """
    
    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Applies tensor conversion.

        Required args
        -------------
        - imgmap: array or list of PIL Images
            Array or list of PIL Images.

        Returns
        -------
        - list of Tensor images
            List of PIL Images converted to Tensor images
        """

        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


#############################################
class Normalize(object):
    """
    Augmentation for normalizing a list of Tensor images.

    Attributes
    ----------
    - mean : iterable
        Normalization mean for each channel.
    - std : iterable
        Normalization standard deviation for each channel.

    Methods
    -------
    - self.__call__(imgmap):
        Normalized a list of Tensor images.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Normalize()

        Constructs a Normalize object.

        Optional args
        -------------
        - mean : iterable (default=[0.485, 0.456, 0.406])
            Normalization mean for each channel.
        - std : iterable (default=[0.229, 0.224, 0.225])
            Normalization standard deviation for each channel.
        """

        self.mean = mean
        self.std = std
    
    def __call__(self, imgmap):
        """
        self.__call__(imgmap)

        Normalizes Tensor images.

        Required args
        -------------
        - imgmap: array or list of Tensor images
            Array or list of Tensor images.

        Returns
        -------
        - list of modified Tensor images
            List of normalized Tensor images
        """

        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


