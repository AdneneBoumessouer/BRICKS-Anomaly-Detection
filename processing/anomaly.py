from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class AnomalyMap:
    "Object to store a labeled image and corresponding anomalies"

    def __init__(self, labeled, regionprops):
        self.labeled = labeled
        self.regionprops = regionprops

    def remove_unsued_labels_from_labeled(self):
        labels_to_keep = [regionprop.label for regionprop in self.regionprops]
        labeled_fil = np.zeros_like(self.labeled)
        for label in labels_to_keep:
            labeled_fil[self.labeled == label] = label
        self.labeled = labeled_fil
        return

    def get_labeled_with_alpha(self, alpha_bg=0.3):
        h, w = self.labeled.shape
        # transform labeled image to rgb image
        labeled_rgb = color.label2rgb(self.labeled, bg_label=0, bg_color=None)
        # add alpha channel to rgb labeled image
        labeled_alpha = np.ones(shape=(h, w, 4), dtype=labeled_rgb.dtype)
        labeled_alpha[:, :, :3] = labeled_rgb
        labeled_alpha[:, :, -1][self.labeled == 0] = alpha_bg
        return labeled_alpha


def generate_anomaly_localization_figure(
    img_input, img_pred, resmap, anomap_lc, anomap_hc, filename
):

    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))
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
    if anomap_lc:
        labeled_alpha = anomap_lc.get_labeled_with_alpha(alpha_bg=0.4)
        axarr[1, 1].imshow(labeled_alpha, alpha=0.7)
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
            axarr[1, 1].set_axis_off()
    axarr[1, 1].set_title("Low Contrast Anomalies")
    axarr[1, 1].set_axis_off()

    axarr[1, 2].imshow(img_input)
    if anomap_hc:
        labeled_alpha = anomap_hc.get_labeled_with_alpha(alpha_bg=0.4)
        axarr[1, 2].imshow(labeled_alpha, alpha=0.7)
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
            axarr[1, 2].set_axis_off()
    axarr[1, 2].set_title("High Contrast Anomalies")
    axarr[1, 2].set_axis_off()
    plt.tight_layout()

    return fig

