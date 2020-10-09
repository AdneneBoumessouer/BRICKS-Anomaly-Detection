import os
import time
import numpy as np
import tensorflow as tf
from processing.resmaps import Resmaps
from processing.utils import printProgressBar, is_rgb
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Postprocessor:
    def __init__(
        self, imgs_input, imgs_pred, filenames, color="grayscale", vmin=0.0, vmax=1.0,
    ):
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        self.filenames = filenames
        self.vmin = vmin
        self.vmax = vmax

        if color == "grayscale":
            self.R_ssim = Resmaps(
                imgs_input=imgs_input,
                imgs_pred=imgs_pred,
                color="grayscale",
                method="ssim",
                filenames=filenames,
                vmin=vmin,
                vmax=vmax,
                dtype="float64",
            )
        else:
            self.R_ssim = Resmaps(
                imgs_input=imgs_input,
                imgs_pred=imgs_pred,
                color="rgb",
                method="mssim",
                filenames=filenames,
                vmin=vmin,
                vmax=vmax,
                dtype="float64",
            )

        self.R_l1 = Resmaps(
            imgs_input=imgs_input,
            imgs_pred=imgs_pred,
            color=color,
            method="l1",
            filenames=filenames,
            vmin=vmin,
            vmax=vmax,
            dtype="float64",
        )

        self.R_l2 = Resmaps(
            imgs_input=imgs_input,
            imgs_pred=imgs_pred,
            color=color,
            method="l2",
            filenames=filenames,
            vmin=vmin,
            vmax=vmax,
            dtype="float64",
        )

    def generate_inspection_figure(self, filenames_plot=[]):
        if filenames_plot != []:
            indicies = [self.filenames.index(filename) for filename in filenames_plot]
        else:
            indicies = list(range(len(self.imgs_input)))

        nrows = len(indicies)
        ncols = 5

        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 5 * nrows))
        for i, j in enumerate(indicies):
            axarr[i, 0].imshow(
                self.imgs_input[j], vmin=self.vmin, vmax=self.vmax, cmap=None
            )
            axarr[i, 0].set_title("Input\n{}".format(self.filenames[j]))
            axarr[i, 0].set_axis_off()

            axarr[i, 1].imshow(
                self.imgs_pred[j], vmin=self.vmin, vmax=self.vmax, cmap=None
            )
            axarr[i, 1].set_title("Reconstruction\n{}".format(self.filenames[j]))
            axarr[i, 1].set_axis_off()

            # resmap ssim gray
            res_ssim = axarr[i, 2].imshow(
                X=self.R_ssim.resmaps[j],
                vmin=self.R_ssim.vmin_resmap,
                vmax=self.R_ssim.vmax_resmap,
                cmap=self.R_ssim.cmap_resmap,
            )
            axarr[i, 2].set_title(
                "Resmap SSIM\n" + f"score = {self.R_ssim.scores[j]:.2E}"
            )
            axarr[i, 2].set_axis_off()
            if self.R_ssim.color == "grayscale":
                fig.colorbar(res_ssim, ax=axarr[i, 2])

            # resmap l1 gray
            res_l1 = axarr[i, 3].imshow(
                X=self.R_l1.resmaps[j],
                vmin=self.R_l1.vmin_resmap,
                vmax=self.R_l1.vmax_resmap,
                cmap=self.R_l1.cmap_resmap,
            )
            axarr[i, 3].set_title("Resmap_L1\n" + f"score = {self.R_l1.scores[j]:.2E}")
            axarr[i, 3].set_axis_off()
            if self.R_l1.color == "grayscale":
                fig.colorbar(res_l1, ax=axarr[i, 3])

            # resmap l2 gray
            res_l2 = axarr[i, 4].imshow(
                X=self.R_l2.resmaps[j],
                vmin=self.R_l2.vmin_resmap,
                vmax=self.R_l2.vmax_resmap,
                cmap=self.R_l2.cmap_resmap,
            )
            axarr[i, 4].set_title("Resmap_L2\n" + f"score = {self.R_l2.scores[j]:.2E}")
            axarr[i, 4].set_axis_off()
            if self.R_l2.color == "grayscale":
                fig.colorbar(res_l2, ax=axarr[i, 4])
            plt.tight_layout()
        return fig


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
