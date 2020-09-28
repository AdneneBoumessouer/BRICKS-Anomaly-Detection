import os
import time
import numpy as np
import tensorflow as tf
from processing.utils import printProgressBar
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Segmentation Parameters
# float + SSIM
THRESH_MIN_FLOAT_SSIM = 0.10
THRESH_STEP_FLOAT_SSIM = 0.002
# float + L2
THRESH_MIN_FLOAT_L2 = 0.005
THRESH_STEP_FLOAT_L2 = 0.0005
# float + ABS
THRESH_MIN_FLOAT_ABS = 0.05
THRESH_STEP_FLOAT_ABS = 0.005
# uint8 + SSIM
THRESH_MIN_UINT8_SSIM = 20
THRESH_STEP_UINT8_SSIM = 1
# uint8 + L2 (generally uneffective combination)
THRESH_MIN_UINT8_L2 = 5
THRESH_STEP_UINT8_L2 = 1
# uint8 + ABS (generally uneffective combination)
THRESH_MIN_UINT8_ABS = 10
THRESH_STEP_UINT8_ABS = 1


class TensorImages:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        vmin,
        vmax,
        color,
        method,
        dtype="float64",
        filenames=None,
    ):
        assert imgs_input.ndim == imgs_pred.ndim == 4
        assert dtype in ["float64", "uint8"]
        assert method in ["l2", "ssim", "mssim", "abs"]
        self.color = color
        self.method = method
        self.dtype = dtype
        # pixel min and max values of input and reconstruction (pred)
        # depend on preprocessing function, which in turn depends on
        # the model used for training.
        self.vmin = vmin
        self.vmax = vmax
        self.filenames = filenames

        # if grayscale, reduce dim to (samples x length x width)
        # if imgs_input.shape[-1] == 1:
        if not is_rgb(imgs_input):
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]
            self.cmap = "gray"
        # if RGB
        else:
            self.cmap = None

        self.cmap_resmap = "inferno" if self.color == "grayscale" else None

        # compute resmaps
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        self.scores, self.resmaps = calculate_resmaps(
            self.imgs_input, self.imgs_pred, color, method, dtype
        )

        # compute maximal threshold based on resmaps
        self.thresh_max = np.amax(self.resmaps)

        # set parameters for future segmentation of resmaps
        if dtype == "float64":
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_FLOAT_SSIM
                self.thresh_step = THRESH_STEP_FLOAT_SSIM
                self.vmin_resmap = 0.0
                self.vmax_resmap = 1.0
            elif method == "l2":
                self.thresh_min = THRESH_MIN_FLOAT_L2
                self.thresh_step = THRESH_STEP_FLOAT_L2
                self.vmin_resmap = 0.0
                self.vmax_resmap = np.amax(self.resmaps)
            elif method == "abs":
                self.thresh_min = THRESH_MIN_FLOAT_ABS
                self.thresh_step = THRESH_STEP_FLOAT_ABS
                self.vmin_resmap = 0.0
                self.vmax_resmap = 1.0

        elif dtype == "uint8":
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_UINT8_SSIM
                self.thresh_step = THRESH_STEP_UINT8_SSIM
                self.vmin_resmap = 0
                self.vmax_resmap = 255
            elif method == "l2":
                self.thresh_min = THRESH_MIN_UINT8_L2
                self.thresh_step = THRESH_STEP_UINT8_L2
                self.vmin_resmap = 0
                self.vmax_resmap = np.amax(self.resmaps)
            elif method == "abs":
                self.thresh_min = THRESH_MIN_UINT8_ABS
                self.thresh_step = THRESH_STEP_UINT8_ABS
                self.vmin_resmap = 0
                self.vmax_resmap = 1.0

    def generate_inspection_plots(self, group, filenames_plot=[], save_dir=None):
        assert group in ["validation", "test"]
        logger.info("generating inspection plots on " + group + " images...")
        if filenames_plot != []:
            indicies = [self.filenames.index(filename) for filename in filenames_plot]
            l = len(filenames_plot)
        else:
            indicies = list(range(len(self.imgs_input)))
            l = len(self.filenames)
        printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=80)
        for j, i in enumerate(indicies):
            self.plot_input_pred_resmap(index=i, group=group, save_dir=save_dir)
            # print progress bar
            time.sleep(0.1)
            printProgressBar(j + 1, l, prefix="Progress:", suffix="Complete", length=80)
        if save_dir is not None:
            logger.info("all generated files are saved at: \n{}".format(save_dir))
        return

    ### plottings methods for inspection

    def generate_inspection_figure(self, filenames_plot=[], threshold=None):
        if filenames_plot != []:
            indicies = [self.filenames.index(filename) for filename in filenames_plot]
        else:
            indicies = list(range(len(self.imgs_input)))

        nrows = len(indicies)
        ncols = 3 if threshold is None else 4

        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
        for i, j in enumerate(indicies):
            axarr[i, 0].imshow(
                self.imgs_input[j], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
            )
            axarr[i, 0].set_title("input")
            axarr[i, 0].set_axis_off()

            axarr[i, 1].imshow(
                self.imgs_pred[j], vmin=self.vmin, vmax=self.vmax, cmap=self.cmap
            )
            axarr[i, 1].set_title("pred")
            axarr[i, 1].set_axis_off()

            res = axarr[i, 2].imshow(
                self.resmaps[j],
                vmin=self.vmin_resmap,
                vmax=self.vmax_resmap,
                cmap=self.cmap_resmap,
            )
            axarr[i, 2].set_title("resmap\n" + f"score = {self.scores[j]:.2E}")
            axarr[i, 2].set_axis_off()
            if self.color == "grayscale":
                fig.colorbar(res, ax=axarr[i, 2])
            if threshold is not None:
                axarr[i, 3].imshow(self.resmaps[j] > threshold)
                axarr[i, 3].set_title("segmentation")
                axarr[i, 3].set_axis_off()
        return

    def generate_score_scatter_plot(self, generator):
        fig = plt.figure(figsize=(15, 8))
        for category in list(generator.class_indices.keys()):
            indicies_cat = np.nonzero(
                generator.classes == generator.class_indices[category]
            )
            scores = self.scores[indicies_cat]
            if category == "good":
                plt.scatter(
                    indicies_cat,
                    scores,
                    alpha=0.5,
                    marker="s",
                    # markersize=6,
                    label=category,
                )
            else:
                plt.scatter(indicies_cat, scores, alpha=0.5, marker=".", label=category)
        plt.xlabel("image index")
        plt.ylabel(self.method + "_score")
        plt.legend()
        return

    def plot_input_pred_resmap(self, index, group, save_dir=None):
        assert group in ["validation", "test"]
        fig, axarr = plt.subplots(3, 1)
        fig.set_size_inches((4, 9))

        axarr[0].imshow(
            self.imgs_input[index], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
        )
        axarr[0].set_title("input")
        axarr[0].set_axis_off()
        # fig.colorbar(im00, ax=axarr[0])

        axarr[1].imshow(
            self.imgs_pred[index], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax
        )
        axarr[1].set_title("pred")
        axarr[1].set_axis_off()
        # fig.colorbar(im10, ax=axarr[1])

        im20 = axarr[2].imshow(
            self.resmaps[index],
            cmap=self.cmap_resmap,
            vmin=self.vmin_resmap,
            vmax=self.vmax_resmap,
        )
        axarr[2].set_title(
            "resmap_"
            + self.method
            + "_"
            + self.dtype
            + "\n{}_".format(self.method)
            + f"score = {self.scores[index]:.2E}"
        )
        axarr[2].set_axis_off()
        if self.color == "grayscale":
            fig.colorbar(im20, ax=axarr[2])

        plt.suptitle(group.upper() + "\n" + self.filenames[index])

        if save_dir is not None:
            plot_name = get_plot_name(self.filenames[index], suffix="inspection")
            fig.savefig(os.path.join(save_dir, plot_name))
            plt.close(fig=fig)
        return

    def plot_image(self, plot_type, index):
        assert plot_type in ["input", "pred", "resmap"]
        # select image to plot
        if plot_type == "input":
            image = self.imgs_input[index]
            cmap = self.cmap
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "pred":
            image = self.imgs_pred[index]
            cmap = self.cmap
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "resmap":
            image = self.resmaps[index]
            cmap = self.cmap_resmap
            vmin = self.vmin_resmap
            vmax = self.vmax_resmap
        # plot image
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        if plot_type == "resmap":
            fig.colorbar(im)
        title = plot_type + "\n" + self.filenames[index]
        plt.title(title)
        plt.show()
        return


def get_plot_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new


#### Image Processing Functions

## Functions for generating Resmaps


def is_rgb(imgs):
    return imgs.ndim == 4 and imgs.shape[-1] == 3


def calculate_resmaps(
    imgs_input, imgs_pred, color_resmap, method_resmap, dtype_resmap="float64"
):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # preprcess according to given parameters
    if color_resmap == "grayscale":
        multichannel = False
        if is_rgb(imgs_input):
            imgs_input_res = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
            imgs_pred_res = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
        else:
            imgs_input_res = imgs_input
            imgs_pred_res = imgs_pred

    elif color_resmap == "rgb":
        if not is_rgb(imgs_input):
            raise ValueError("imgs_input and imgs_pred have to be rgb.")
        multichannel = True
        imgs_input_res = imgs_input
        imgs_pred_res = imgs_pred

    # calculate remaps
    if method_resmap == "l2":
        scores, resmaps = resmaps_l2(imgs_input_res, imgs_pred_res)
    elif method_resmap in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_res, imgs_pred_res, multichannel)
    elif method_resmap == "abs":
        scores, resmaps = resmaps_abs(imgs_input_res, imgs_pred_res)

    # convert to uint8 if mentioned
    if dtype_resmap == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def resmaps_ssim(imgs_input, imgs_pred, multichannel):
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
            multichannel=multichannel,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    scores = np.array(scores)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    # scores = np.sqrt(np.sum(resmaps, axis=0)).flatten()
    scores = np.sum(resmaps, axis=0).flatten()
    return scores, resmaps


def resmaps_abs(imgs_input, imgs_pred):
    resmaps = np.abs(imgs_input - imgs_pred)
    scores = np.sum(resmaps, axis=0).flatten()
    return scores, resmaps


## functions for processing resmaps


def label_images(images_th):
    """
    Segments images into images of connected components (regions).
    Returns segmented images and a list of lists, where each list 
    contains the areas of the regions of the corresponding image. 
    
    Parameters
    ----------
    images_th : array of binary images
        Thresholded residual maps.

    Returns
    -------
    images_labeled : array of labeled images
        Labeled images.
    areas_all : list of lists
        List of lists, where each list contains the areas of the regions of the corresponding image.

    """
    images_labeled = np.zeros(shape=images_th.shape)
    areas_all = []
    for i, image_th in enumerate(images_th):
        # close small holes with binary closing
        # bw = closing(image_th, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(image_th)

        # label image regions
        image_labeled = label(cleared)

        # image_labeled = label(image_th)

        # append image
        images_labeled[i] = image_labeled

        # compute areas of anomalous regions in the current image
        regions = regionprops(image_labeled)

        if regions:
            areas = [region.area for region in regions]
            areas_all.append(areas)
        else:
            areas_all.append([0])

    return images_labeled, areas_all

