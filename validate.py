import os
import argparse
import json
import numpy as np
from skimage import measure
from processing.preprocessing import Preprocessor
from processing import utils
from processing.utils import printProgressBar
from processing.resmaps import ResmapCalculator
import config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_threshold(resmaps_val, min_area=25):
    # initialize thresholds
    th_max = np.amax(resmaps_val)
    th_min = np.amin(resmaps_val)
    th_step = 5e-3
    ths = np.arange(start=th_min, stop=th_max, step=th_step, dtype="float")

    printProgressBar(0, len(ths), prefix="Progress:", suffix="Complete", length=80)

    # loop over thresholds
    for i, th in enumerate(ths):
        imgs_binary = resmaps_val > th
        imgs_labeled = np.array([measure.label(binary) for binary in imgs_binary])
        areas = np.array(
            [
                regionprop.area
                for labeled in imgs_labeled
                for regionprop in measure.regionprops(labeled)
            ]
        )
        largest_area = np.amax(areas)
        if largest_area < min_area:
            break
        printProgressBar(
            i + 1, len(ths), prefix="Progress:", suffix="Complete", length=80
        )

    printProgressBar(
        len(ths), len(ths), prefix="Progress:", suffix="Complete", length=80
    )
    return th


def validate(model_path, method):
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

    val_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )
    # imgs_val_input = val_generator.next()[0]

    # retrieve validation images for front view a00
    # TODO unelegant code: generator.filenames scanned two times
    views = ["a00", "a45"]
    for view in views:
        filenames = [
            filename
            for filename in val_generator.filenames
            if filename.split("/")[-1].split("_")[0] == view
        ]
        index_array = [
            i
            for i, filename in enumerate(val_generator.filenames)
            if filename.split("/")[-1].split("_")[0] == view
        ]
        imgs_val_input = val_generator._get_batches_of_transformed_samples(index_array)[
            0
        ]

        # reconstruct validation inspection images (i.e predict)
        imgs_val_pred = model.predict(imgs_val_input)

        # calculate resmaps
        RC_val_a00 = ResmapCalculator(
            imgs_input=imgs_val_input,
            imgs_pred=imgs_val_pred,
            color_out="grayscale",
            method=method,
            filenames=filenames,
            vmin=vmin,
            vmax=vmax,
            dtype="float64",
        )
        resmaps_val = RC_val_a00.get_resmaps()

        if view == "a00":
            min_area = config.MIN_AREA_VAL_a00
        else:
            min_area = config.MIN_AREA_VAL_a45

        # estimate threshold
        th = estimate_threshold(resmaps_val, min_area=min_area)

        # save validation results
        validation_result = {"min_area": min_area, "th": th}

        # save test result
        save_dir = os.path.join(os.path.dirname(model_path), "validation", view)
        if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "result_" + view + ".json"), "w") as json_file:
            json.dump(validation_result, json_file, indent=4, sort_keys=False)
    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Test model on some images for inspection.",
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
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

    # parse arguments
    args = parser.parse_args()

    # run main function
    validate(model_path=args.path, method=args.method)

# python3 validate.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 --method l1
