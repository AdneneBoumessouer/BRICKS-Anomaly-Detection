import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from skimage import measure
from skimage import color
from processing.preprocessing import Preprocessor
from processing.resmaps import ResmapCalculator
from processing import utils
from processing.utils import printProgressBar
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_labeled_imgs(
    resmap, filename, min_area_plot=5, alpha_bg=0.3, save_dir=None
):
    """
    Generates labeled images for increasing thresholds cooresponding
    to a given resmap containing bboxes for regions which areas are bigger
    than min_area_plot.
    Generated images are saved in a directory and demonstrate how the validation
    algorithm works. It also helps to estimate visually a good min_area to be used 
    for estimating threshold.
    """
    # initialize thresholds
    th_min = 0.1 # np.amin(resmap) + th_step
    th_step = 2e-3
    th_max = np.amax(resmap) + th_step
    ths = np.arange(start=th_min, stop=th_max, step=th_step, dtype="float")

    logger.info("generating labeled resmaps for {}".format(filename))
    printProgressBar(0, len(ths), prefix="Progress:", suffix="Complete", length=80)

    # label resmaps with increasing thresholds
    for i, th in enumerate(ths):
        binary = resmap > th
        labeled = measure.label(binary)
        regionprops = measure.regionprops(labeled)
        regionprops.sort(key=lambda x: x.area, reverse=True)

        props = [(regionprop.area, regionprop.bbox) for regionprop in regionprops]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # transform labeled image to rgb image
        labeled_rgb = color.label2rgb(labeled, bg_label=0, bg_color=None)
        # add alpha channel to rgb labeled image
        h, w = labeled.shape
        labeled_alpha = np.ones(shape=(h, w, 4), dtype=labeled_rgb.dtype)
        labeled_alpha[:, :, :3] = labeled_rgb
        labeled_alpha[:, :, -1][labeled == 0] = alpha_bg

        ax.imshow(resmap, vmin=0.0, vmax=1.0, cmap="inferno")
        ax.imshow(labeled_alpha, alpha=0.5)
        if props:
            for area, bbox in props:
                if area > min_area_plot:
                    # edgecolor = "red" if area >= min_area else "yellow"
                    minr, minc, maxr, maxc = bbox
                    rect = mpatches.Rectangle(
                        (minc, minr),
                        maxc - minc,
                        maxr - minr,
                        fill=False,
                        edgecolor="yellow",  # edgecolor
                        linewidth=1.0,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x=minc,
                        y=minr - 5,
                        s=str(area),
                        verticalalignment="top",
                        horizontalalignment="left",
                        color="yellow",  # edgecolor
                        fontweight="bold",
                        fontsize=10,
                    )
        ax.set_axis_off()
        ax.set_title("Labeled Resmap of: {}\nthreshold = {:.3}".format(filename, th))
        fig.savefig(os.path.join(save_dir, "{}.png".format(i)))
        plt.close(fig)

        printProgressBar(
            i + 1, len(ths), prefix="Progress:", suffix="Complete", length=80
        )
    return


def main(model_path, method, subset, view):
    # load model for inspection
    logger.info("loading model for inspection...")
    model, info, _ = utils.load_model_HDF5(model_path)
    input_dir = info["data"]["input_directory"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    vmin = info["preprocessing"]["vmin"]
    vmax = info["preprocessing"]["vmax"]
    nb_validation_images = info["data"]["nb_validation_images"]

    # instantiate preprocessor object to preprocess validation and test inspection images
    preprocessor = Preprocessor(
        input_directory=input_dir, rescale=rescale, shape=shape, color_mode=color_mode,
    )
    if subset == "val":
        generator = preprocessor.get_val_generator(
            batch_size=nb_validation_images, shuffle=False
        )
        filenames_insp = config.FILENAMES_VAL_INSPECTION
    else:
        nb_test_images = preprocessor.get_total_number_test_images()
        generator = preprocessor.get_test_generator(
            batch_size=nb_test_images, shuffle=False
        )
        filenames_insp = config.FILENAMES_TEST_INSPECTION

    # get index array for filenames to label
    filenames = [
        filename
        for filename in filenames_insp
        if filename.split("/")[-1].split("_")[0] == view
    ]
    arr_i = [
        generator.filenames.index(filename)
        for filename in filenames_insp
        if filename.split("/")[-1].split("_")[0] == view
    ]
    # TODO FIX: get_val_generator() with test images ambegious

    imgs_input = generator.next()[0][arr_i]
    imgs_pred = model.predict(imgs_input)

    RC_val = ResmapCalculator(
        imgs_input=imgs_input,
        imgs_pred=imgs_pred,
        color_out="grayscale",
        method=method,
        filenames=filenames,
        vmin=vmin,
        vmax=vmax,
    )
    resmaps = RC_val.get_resmaps()

    for resmap, filename in list(zip(resmaps, filenames)):
        # create save dir
        save_dir = os.path.join(
            os.path.dirname(model_path), "labeling", method, subset, view, filename
        )
        os.makedirs(save_dir, exist_ok=True))
        # generate and save labeled resmaps for increasing thresholds
        generate_labeled_imgs(resmap, filename, min_area_plot=5, save_dir=save_dir)
    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Test model on some images for inspection.",
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    # TODO add color_out in args
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=False,
        metavar="",
        choices=["ssim", "l1", "l2", "combined"],
        default="l1",
        help="method used to compute resmaps",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=str,
        required=True,
        metavar="",
        choices=["val", "test"],
        default="val",
        help="subset of inspection images (in config.py) to label",
    )
    parser.add_argument(
        "-v",
        "--view",
        type=str,
        required=True,
        metavar="",
        choices=["a00", "a45"],
        help="view dataset to perform classification on",
    )

    # parse arguments
    args = parser.parse_args()

    # run main function
    main(model_path=args.path, method=args.method, subset=args.subset, view=args.view)


# python3 label.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 --method l2 --subset val --view a00
