import os
import numpy as np
import tensorflow as tf
from processing.utils import printProgressBar, is_rgb
from skimage.metrics import structural_similarity
from skimage.util import img_as_ubyte
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResmapCalculator:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        color,
        method,
        filenames=None,
        vmin=0.0,
        vmax=1.0,
        dtype="float64",
    ):
        assert imgs_input.ndim == imgs_pred.ndim == 4
        assert method in ["l1", "l2", "ssim", "mssim"]
        assert color in ["grayscale", "rgb"]
        assert dtype in ["float64", "uint8"]

        if method == "mssim" and (not is_rgb(imgs_input) or color == "grayscale"):
            raise ValueError("mssim method incompatible with grayscale")

        self.color = color
        self.method = method
        self.dtype = dtype
        self.filenames = filenames

        # pixel min and max values of input and reconstruction (pred)
        # depend on preprocessing function, which in turn depends on
        # the model used for training.
        self.vmin = vmin
        self.vmax = vmax

        # if grayscale, reduce dim to (samples x length x width)
        if imgs_input.ndim == 4 and imgs_input.shape[-1] == 1:
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]

        self.cmap_resmap = "inferno" if self.color == "grayscale" else None

        # compute resmaps
        self.scores, self.resmaps = calculate_resmaps(
            imgs_input, imgs_pred, color, method, dtype
        )

        # set parameters for future segmentation of resmaps
        self.vmin_resmap = 0
        self.vmax_resmap = np.amax(self.resmaps)

    def get_resmaps(self):
        return self.resmaps

    def get_scores(self):
        return self.scores


#### Functions to calculate Resmaps


def calculate_resmaps(
    imgs_input, imgs_pred, color_resmap, method_resmap, dtype_resmap="float64"
):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # preprcess according to given parameters
    if color_resmap == "grayscale":
        # multichannel = False
        if is_rgb(imgs_input):
            imgs_input_res = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
            imgs_pred_res = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
        else:
            imgs_input_res = imgs_input
            imgs_pred_res = imgs_pred

    elif color_resmap == "rgb":
        if not is_rgb(imgs_input):
            raise ValueError("input images are not rgb.")
        # multichannel = True
        imgs_input_res = imgs_input
        imgs_pred_res = imgs_pred

    # calculate remaps
    if method_resmap == "ssim":
        scores, resmaps = resmaps_ssim(imgs_input_res, imgs_pred_res)
    elif method_resmap == "mssim":
        scores, resmaps = resmaps_mssim(imgs_input_res, imgs_pred_res)
    elif method_resmap == "l1":
        scores, resmaps = resmaps_l1(imgs_input_res, imgs_pred_res)
    elif method_resmap == "l2":
        scores, resmaps = resmaps_l2(imgs_input_res, imgs_pred_res)

    # convert to uint8 if mentioned
    if dtype_resmap == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def resmaps_ssim(imgs_input, imgs_pred):
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
            multichannel=True,
            sigma=1.5,
            full=True,
        )
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    scores = np.array(scores)
    return scores, resmaps


def resmaps_mssim(imgs_input, imgs_pred):
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
            multichannel=False,
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

