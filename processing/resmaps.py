import os
import numpy as np
import tensorflow as tf
from processing.utils import printProgressBar, is_rgb
from skimage.metrics import structural_similarity
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResmapCalculator:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        color_out,
        method,
        filenames=None,
        vmin=0.0,
        vmax=1.0,
        dtype="float64",  # TODO remove support for uint8 and continue only with float64
    ):
        assert imgs_input.ndim == imgs_pred.ndim == 4
        assert method in ["l1", "l2", "ssim", "combined"]
        assert color_out in ["grayscale", "rgb"]
        assert dtype in ["float64", "uint8"]

        if color_out == "rgb" and not(is_rgb(imgs_input)):
            raise ValueError("color is rgb but images are grayscale")

        self.color_out = color_out
        self.method = method
        self.dtype = dtype
        self.filenames = filenames

        # min and max pixel values of input and reconstruction (pred).
        # They depend on preprocessing function used by the model during training.
        self.vmin = vmin
        self.vmax = vmax

        # if grayscale, reduce dim to (samples x length x width)
        if imgs_input.ndim == 4 and imgs_input.shape[-1] == 1:
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]

        self.cmap_resmap = "inferno" if color_out == "grayscale" else None

        # compute resmaps
        self.scores, self.resmaps = calculate_resmaps(
            imgs_input, imgs_pred, color_out, method, dtype
        )

        # set parameters for future segmentation of resmaps
        self.vmin_resmap = 0
        self.vmax_resmap = np.amax(self.resmaps)  # TODO test with 1.0

    def get_resmaps(self):
        return self.resmaps

    def get_scores(self):
        return self.scores


# Functions to calculate Resmaps


def calculate_resmaps(
    imgs_input, imgs_pred, color_out, method, dtype="float64"
):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # convert to grayscale
    if color_out == "grayscale" and is_rgb(imgs_input):
        imgs_input = rgb2gray(imgs_input)
        imgs_pred = rgb2gray(imgs_pred)

    # calculate remaps
    if method == "ssim":
        scores, resmaps = resmaps_ssim(imgs_input, imgs_pred)
    elif method == "l1":
        scores, resmaps = resmaps_l1(imgs_input, imgs_pred)
    elif method == "l2":
        scores, resmaps = resmaps_l2(imgs_input, imgs_pred)
    elif method == "combined":
        scores, resmaps = resmaps_combined(imgs_input, imgs_pred)
    # change dtype to unsigned integer
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def resmaps_ssim(imgs_input, imgs_pred):
    # chaek channels
    if is_rgb(imgs_input):
        multichannel = True
    else:
        multichannel = False

    # calculate resmaps
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=multichannel,  # True
            sigma=1.5,
            full=True,
        )
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    scores = np.array(scores)
    return scores, resmaps


def resmaps_l1(imgs_input, imgs_pred):
    resmaps = np.abs(imgs_input - imgs_pred)
    scores = np.sum(resmaps, axis=0).flatten()
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = np.sum(resmaps, axis=0).flatten()
    return scores, resmaps


def resmaps_combined(imgs_input, imgs_pred):
    resmaps = 0.7*resmaps_ssim(imgs_input, imgs_pred) + \
        1.0*resmaps_l1(imgs_input, imgs_pred)
    scores = np.sum(resmaps, axis=0).flatten()
    return scores, resmaps
