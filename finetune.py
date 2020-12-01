import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from processing import utils
from processing.resmaps import ResmapCalculator
from processing.preprocessing import Preprocessor
from validate import estimate_threshold
from test import predict_classes
from processing.utils import printProgressBar
from sklearn.metrics import confusion_matrix

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_MIN_AREA = 5
STEP_MIN_AREA = 5
STOP_MIN_AREA = 1205


def finetune(model_path, view, method):
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

    # views = ["a00", "a45"]
    # for view in views:
    filenames_val = [
        filename
        for filename in val_generator.filenames
        if filename.split("/")[-1].split("_")[0] == view
    ]
    index_array_val = [
        i
        for i, filename in enumerate(val_generator.filenames)
        if filename.split("/")[-1].split("_")[0] == view
    ]
    imgs_val_input = val_generator._get_batches_of_transformed_samples(index_array_val)[
        0
    ]

    # reconstruct validation inspection images (i.e predict)
    imgs_val_pred = model.predict(imgs_val_input)

    # calculate resmaps
    RC_val = ResmapCalculator(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        color_out="grayscale",
        method=method,
        filenames=filenames_val,
        vmin=vmin,
        vmax=vmax,
        dtype="float64",
    )
    resmaps_val = RC_val.get_resmaps()

    # -------------------------------------------------------------------

    # get test generator
    nb_test_images = preprocessor.get_total_number_test_images()
    test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )

    filenames_test = [
        filename
        for filename in test_generator.filenames
        if filename.split("/")[-1].split("_")[0] == view
    ]
    index_array_test = [
        i
        for i, filename in enumerate(test_generator.filenames)
        if filename.split("/")[-1].split("_")[0] == view
    ]
    imgs_test_input = test_generator._get_batches_of_transformed_samples(
        index_array_test
    )[0]

    # reconstruct validation inspection images (i.e predict)
    imgs_test_pred = model.predict(imgs_test_input)

    # calculate resmaps
    RC_test = ResmapCalculator(
        imgs_input=imgs_test_input,
        imgs_pred=imgs_test_pred,
        color_out="grayscale",
        method=method,
        filenames=filenames_test,
        vmin=vmin,
        vmax=vmax,
        dtype="float64",
    )
    resmaps_test = RC_test.get_resmaps()

    # retrieve ground truth
    y_true = [0 if "good" in filename.split("/") else 1 for filename in filenames_test]

    # ======================== COMPUTE THRESHOLDS ===========================
    # initialize finetuning dictionary
    stats = {
        "min_area": [],
        "threshold": [],
        "TPR": [],
        "TNR": [],
        "score": [],
    }

    # initialize discrete min_area values
    min_areas = np.arange(start=START_MIN_AREA, stop=STOP_MIN_AREA, step=STEP_MIN_AREA)

    printProgressBar(0, len(min_areas), length=80)

    for i, min_area in enumerate(min_areas):
        th = estimate_threshold(resmaps_val, min_area, verbose=0)
        y_pred = predict_classes(resmaps_test, min_area, th)
        tnr, _, _, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

        # record current results
        stats["min_area"].append(int(min_area))
        stats["threshold"].append(float(th))
        stats["TPR"].append(float(tpr))
        stats["TNR"].append((float(tnr)))
        stats["score"].append(0.7 * float(tpr) + 0.3 * float(tnr))

        printProgressBar(i + 1, len(min_areas), length=80)

    # get min_area, threshold pair corresponding to best score
    max_score_i = np.argmax(stats["score"])
    max_score = stats["score"][max_score_i]
    best_min_area = stats["min_area"][max_score_i]
    best_threshold = stats["threshold"][max_score_i]

    best_stats = {
        "best_min_area": best_min_area,
        "best_threshold": best_threshold,
        "best_score": max_score,
        "method": method,
    }

    # ===================== SAVE FINETUNING RESULTS ========================
    # save finetuning result
    save_dir = os.path.join(os.path.dirname(model_path), "finetuning", view)
    if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
        os.makedirs(save_dir)
    pd.DataFrame.from_dict(stats).to_pickle(os.path.join(save_dir, "df_stats.pkl"))
    with open(os.path.join(save_dir, "stats.json"), "w") as json_file:
        json.dump(stats, json_file, indent=4, sort_keys=False)
    with open(os.path.join(save_dir, "best_stats.json"), "w") as json_file:
        json.dump(best_stats, json_file, indent=4, sort_keys=False)

    # save finetuning plots
    fig1 = plot_min_area_threshold(stats, index_best=max_score_i)
    fig1.savefig(os.path.join(save_dir, "min_area vs. threshold.png"))
    fig2 = plot_stats(stats, index_best=max_score_i)
    fig2.savefig(os.path.join(save_dir, "stats.png"))
    plt.close("all")
    return


def plot_min_area_threshold(stats, index_best):
    with plt.style.context("seaborn-darkgrid"):
        # plot min_area vs. threshold
        fig = plt.figure(figsize=(16, 9))
        plt.plot(stats["min_area"], stats["threshold"])
        plt.xlabel("min areas")
        plt.ylabel("thresholds")
        # plot best (mon_area, threshold) pair
        x = stats["min_area"][index_best]
        y = stats["threshold"][index_best]
        plt.axvline(x, 0, y, linestyle="dashed", color="red", linewidth=0.5)
        plt.axhline(y, 0, x, linestyle="dashed", color="red", linewidth=0.5)
        label_marker = "best min_area / threshold pair"
        plt.plot(x, y, markersize=10, marker="o", color="red", label=label_marker)
        plt.title(
            "min_area vs. threshold\nbest min_area = {}\nbest threshold = {:.4f}".format(
                x, y
            )
        )
    return fig


def plot_stats(stats, index_best):
    with plt.style.context("seaborn-darkgrid"):
        fig = plt.figure(figsize=(16, 9))
        plt.plot(stats["min_area"], stats["TPR"], label="TPR")
        plt.plot(stats["min_area"], stats["TNR"], label="TNR")
        plt.plot(stats["min_area"], stats["score"], label="score")
        plt.xlabel("min areas")
        plt.ylabel("stats")
        # plot best (mon_area, threshold) pair
        x = stats["min_area"][index_best]
        y = stats["score"][index_best]
        plt.axvline(x, 0, 1, linestyle="dashed", color="red", linewidth=0.5)
        plt.plot(x, y, markersize=10, marker="o", color="red", label="best score")
        plt.title(f"Stats Plot\nbest score = {y:.2E}")
        plt.legend()
    return fig


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Test model on some images for inspection.",
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
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
    finetune(model_path=args.path, view=args.view, method=args.method)

# Example of command to initiate finetuning with different resmap processing arguments (best combination: -m ssim -t float64)

# python3 finetune.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 -v a00 -m l1