from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from processing import utils
from skimage import util


class AnomalyMap:
    "Object to store a labeled image and corresponding anomalies"

    def __init__(self, labeled, regionprops):
        # filtered labeled image
        self.labeled = labeled
        # only defective regionprops
        self.regionprops = regionprops

    def __bool__(self):
        if self.regionprops:
            return True
        return False

    def get_mask(self):
        mask = self.labeled > 0
        return mask

    def get_labeled_with_alpha(self, alpha_bg=0.3):
        h, w = self.labeled.shape
        # transform labeled image to rgb image
        labeled_rgb = color.label2rgb(self.labeled, bg_label=0, bg_color=None)
        # add alpha channel to rgb labeled image
        labeled_alpha = np.ones(shape=(h, w, 4), dtype=labeled_rgb.dtype)
        labeled_alpha[:, :, :3] = labeled_rgb
        labeled_alpha[:, :, -1][self.labeled == 0] = alpha_bg
        return labeled_alpha


def filter_labeled(labeled, regionprops):
    """
    retains labeled pixels in @param:labeled which labels are contained in regionprops.
    """
    labeled_fil = np.zeros_like(labeled)
    if regionprops:
        labels_to_keep = [regionprop.label for regionprop in regionprops]
        for label in labels_to_keep:
            labeled_fil[labeled == label] = label
    return labeled_fil


def get_labeled_with_alpha(labeled, colors=["red"], alpha_bg=0.3):
    h, w = labeled.shape
    # transform labeled image to rgb image
    labeled_rgb = color.label2rgb(labeled, colors=colors, bg_label=0, bg_color=None)
    # add alpha channel to rgb labeled image
    labeled_alpha = np.ones(shape=(h, w, 4), dtype=labeled_rgb.dtype)
    labeled_alpha[:, :, :3] = labeled_rgb
    labeled_alpha[:, :, -1][labeled == 0] = alpha_bg
    return labeled_alpha


def merge_labeled(anomap_lc, anomap_hc, shape=(256, 256)):
    labeled_merged = np.zeros(shape, dtype="int64")
    if anomap_lc:
        labeled_merged[anomap_lc.labeled > 0] = 1
    if anomap_hc:
        labeled_merged[anomap_hc.labeled > 0] = 2
    return labeled_merged


def generate_anomaly_localization_figure(
    img_input, img_pred, resmap, anomap_lc, anomap_hc, filename
):
    nrows, ncols = 2, 3
    figsize = utils.get_optimal_figsize(nrows, ncols)

    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(filename, fontsize=20)

    axarr[0, 0].imshow(img_input)
    axarr[0, 0].set_title("input")
    axarr[0, 0].set_axis_off()

    axarr[0, 1].imshow(img_pred)
    axarr[0, 1].set_title("Reconstruction")
    axarr[0, 1].set_axis_off()

    axarr[0, 2].set_axis_off()

    res = axarr[1, 0].imshow(resmap, cmap="inferno", vmin=0.0, vmax=1.0)
    axarr[1, 0].set_title("Resmap")
    axarr[1, 0].set_axis_off()
    fig.colorbar(res, ax=axarr[1, 0])

    axarr[1, 1].imshow(img_input)
    axarr[1, 1].set_title("Low Contrast Anomalies")
    axarr[1, 1].set_axis_off()
    if anomap_lc:
        labeled_alpha = anomap_lc.get_labeled_with_alpha(alpha_bg=0.4)
        axarr[1, 1].imshow(labeled_alpha, alpha=0.7)
        axarr[1, 1].set_axis_off()
        for regionprop in anomap_lc.regionprops:
            minr, minc, maxr, maxc = regionprop.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5,
            )
            axarr[1, 1].add_patch(rect)
            axarr[1, 1].text(
                x=minc,
                y=minr - 7,
                s=str(regionprop.area),
                verticalalignment="top",
                horizontalalignment="left",
                color="yellow",
                fontweight="bold",
                fontsize=10,
            )

    axarr[1, 2].imshow(img_input)
    axarr[1, 2].set_title("High Contrast Anomalies")
    axarr[1, 2].set_axis_off()
    if anomap_hc:
        labeled_alpha = anomap_hc.get_labeled_with_alpha(alpha_bg=0.4)
        axarr[1, 2].imshow(labeled_alpha, alpha=0.7)
        axarr[1, 2].set_axis_off()
        for regionprop in anomap_hc.regionprops:
            minr, minc, maxr, maxc = regionprop.bbox
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
            axarr[1, 2].add_patch(rect)
            axarr[1, 2].text(
                x=minc,
                y=minr - 7,
                s=str(regionprop.area),
                verticalalignment="top",
                horizontalalignment="left",
                color="red",
                fontweight="bold",
                fontsize=10,
            )

    plt.tight_layout()

    return fig


def generate_segmentation_figure(
    img_input, img_pred, resmap, mask_true, anomap_lc, anomap_hc, filename
):

    nrows, ncols = 2, 3
    figsize = utils.get_optimal_figsize(nrows, ncols)

    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(filename, fontsize=20)

    axarr[0, 0].imshow(img_input)
    axarr[0, 0].set_title("input")
    axarr[0, 0].set_axis_off()

    axarr[0, 1].imshow(img_pred)
    axarr[0, 1].set_title("Reconstruction")
    axarr[0, 1].set_axis_off()
    axarr[0, 2].set_axis_off()

    res = axarr[1, 0].imshow(resmap, cmap="inferno", vmin=0.0, vmax=1.0)
    axarr[1, 0].set_title("Resmap")
    axarr[1, 0].set_axis_off()
    fig.colorbar(res, ax=axarr[1, 0])

    axarr[1, 1].imshow(img_input)
    axarr[1, 1].set_axis_off()
    axarr[1, 1].set_title("Anomaly Map")

    # compute IoU
    if np.any(mask_true):
        mask_pred = anomap_lc.get_mask() | anomap_hc.get_mask()
        IoU = calculate_IoU(mask_true, mask_pred)
        title_seg = "Segmentation Map\nIoU = {:.3f}".format(IoU)
    else:
        title_seg = "Segmentation Map"

    axarr[1, 2].imshow(mask_true, cmap="gray")
    axarr[1, 2].set_title(title_seg)
    axarr[1, 2].set_axis_off()

    if anomap_lc or anomap_hc:
        colors = []
        if anomap_lc:
            colors.append("yellow")
        if anomap_hc:
            colors.append("red")
        labeled_merged = merge_labeled(anomap_lc, anomap_hc)
        labeled_alpha_anomap = get_labeled_with_alpha(
            labeled_merged, colors=colors, alpha_bg=0.4
        )
        labeled_alpha_mask = get_labeled_with_alpha(
            labeled_merged, colors=colors, alpha_bg=0.0
        )
        axarr[1, 1].imshow(labeled_alpha_anomap, alpha=0.7)
        axarr[1, 2].imshow(labeled_alpha_mask, alpha=0.7)

    plt.tight_layout()
    return fig


def calculate_IoU(mask_true, mask_pred):
    # convert inputs to boolean type if necessary
    if not mask_true.dtype == np.dtype("bool"):
        mask_true = util.img_as_bool(mask_true)
    if not mask_pred.dtype == np.dtype("bool"):
        mask_pred = util.img_as_bool(mask_pred)
    # initialize masks
    mask_inter = np.zeros(shape=mask_true.shape, dtype="bool")
    mask_union = np.zeros(shape=mask_true.shape, dtype="bool")
    # compute intersection and union masks
    mask_inter[(mask_true == True) & (mask_pred == True)] = True
    mask_union[(mask_true == True) | (mask_pred == True)] = True
    # compute IoU: Intersection over Union
    if np.count_nonzero(mask_union) == 0:
        IoU = 1.0
    else:
        IoU = np.count_nonzero(mask_inter) / np.count_nonzero(mask_union)
    return IoU
