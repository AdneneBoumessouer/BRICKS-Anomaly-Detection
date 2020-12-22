import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from processing import utils
from processing.resmaps import ResmapCalculator
from processing.preprocessing import Preprocessor
from processing import detection
from processing.utils import printProgressBar
from sklearn.metrics import confusion_matrix

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_MIN_AREA_HC = 5
STEP_MIN_AREA_HC = 5
STOP_MIN_AREA_HC = 200


def main(model_path, view, method):
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

    # -------------------------------------------------------------------

    val_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )
    index_array_val, filenames_val = utils.get_indices(val_generator, view)
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
    )
    resmaps_val = RC_val.get_resmaps()

    # -------------------------------------------------------------------

    # get test generator
    nb_test_images = preprocessor.get_total_number_test_images()
    test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )

    # test_generator.reset
    categories = ["02_added", "03_missing", "04_shifted", "10_stain", "good"]
    index_array_test, filenames_test = utils.get_indices(
        test_generator, view, categories
    )
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
    )
    resmaps_test = RC_test.get_resmaps()

    # ======================== COMPUTE THRESHOLDS ===========================

    # retrieve ground truth
    y_true = [0 if "good" in filename.split("/") else 1 for filename in filenames_test]

    # initialize finetuning dictionary
    stats = {
        "min_area_hc": [],
        "threshold": [],
        "min_area_lc": [],
        "TPR": [],
        "FPR": [],
        "TNR": [],
        "FNR": [],
        "score": [],
    }

    # initialize discrete min_area values
    min_areas_hc = np.arange(
        start=START_MIN_AREA_HC, stop=STOP_MIN_AREA_HC, step=STEP_MIN_AREA_HC
    )
    n = len(min_areas_hc)
    printProgressBar(0, n, prefix="Progress:", suffix="Complete", length=80)

    for i, min_area_hc in enumerate(min_areas_hc):
        printProgressBar(
            iteration=i + 1,
            total=n,
            prefix="Progress:",
            suffix="Complete | Current min_area_hc = {}".format(min_area_hc),
            length=80,
        )

        # initialize detectors
        detector_hc = detection.HighContrastAnomalyDetector()
        detector_lc = detection.LowContrastAnomalyDetector()

        # estimate threshold for high contrast anomalies
        threshold = detector_hc.estimate_threshold(resmaps_val, min_area_hc, verbose=0)

        # break out of loop if threshold inferior to vmin
        if threshold < detector_lc.vmin:
            print()
            break

        # estimate area for low contrast anomalies
        min_area_lc = detector_lc.estimate_area(resmaps_val, threshold)

        # predict
        y_pred_hc, _ = detector_hc.predict(resmaps_test)
        y_pred_lc, _ = detector_lc.predict(resmaps_test)
        y_pred = list(np.array(y_pred_lc) | np.array(y_pred_hc))

        # get results
        tnr, fpr, fnr, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

        # record current results
        stats["min_area_hc"].append(int(min_area_hc))
        stats["threshold"].append(float(threshold))
        stats["min_area_lc"].append(int(min_area_lc))
        stats["TPR"].append(float(tpr))
        stats["FPR"].append(float(fpr))
        stats["TNR"].append(float(tnr))
        stats["FNR"].append(float(fnr))
        stats["score"].append(0.5 * float(tpr) + 0.5 * float(tnr))

    # get min_area, threshold pair corresponding to best score
    index_best_score = np.argmax(stats["score"])

    best_stats = {
        "best_min_area_hc": stats["min_area_hc"][index_best_score],
        "best_threshold": stats["threshold"][index_best_score],
        "best_min_area_lc": stats["min_area_lc"][index_best_score],
        "best_score": stats["score"][index_best_score],
        "best_TPR": stats["TPR"][index_best_score],
        "best_TNR": stats["TNR"][index_best_score],
        "method": method,
    }

    # ===================== SAVE FINETUNING RESULTS ========================
    # save finetuning result
    save_dir = os.path.join(os.path.dirname(model_path), "finetuning", method, view)
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame.from_dict(stats).to_pickle(os.path.join(save_dir, "df_stats.pkl"))
    with open(os.path.join(save_dir, "best_stats.json"), "w") as json_file:
        json.dump(best_stats, json_file, indent=4, sort_keys=False)

    # save finetuning plots
    fig1 = plot_stats(stats, index_best_score=index_best_score)
    fig1.savefig(os.path.join(save_dir, "stats.png"))
    fig2 = plot_roc(stats)
    fig2.savefig(os.path.join(save_dir, "roc.png"))
    plt.close("all")
    return


def plot_stats(stats, index_best_score):
    with plt.style.context("seaborn-darkgrid"):
        # plot min_area vs. threshold
        fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        axarr[0].plot(stats["min_area_hc"], stats["threshold"])
        axarr[0].set_xlabel("min areas_hc")
        axarr[0].set_ylabel("thresholds_hc")
        # plot best (mon_area, threshold) pair
        x = stats["min_area_hc"][index_best_score]
        y = stats["threshold"][index_best_score]
        axarr[0].axvline(x, 0, y, linestyle="dashed", color="red", linewidth=0.5)
        axarr[0].axhline(y, 0, x, linestyle="dashed", color="red", linewidth=0.5)
        label_marker = "best min_area / threshold pair"
        axarr[0].plot(x, y, markersize=5, marker="o", color="red", label=label_marker)
        axarr[0].set_title(
            "min_area_hc vs. threshold\nbest min_area_hc = {} | best threshold = {:.4f}".format(
                x, y
            )
        )
        # plot stats
        axarr[1].plot(stats["min_area_hc"], stats["TPR"], label="TPR")
        axarr[1].plot(stats["min_area_hc"], stats["TNR"], label="TNR")
        axarr[1].plot(stats["min_area_hc"], stats["score"], label="score")
        axarr[1].set_xlabel("min areas_hc")
        axarr[1].set_ylabel("stats")
        # plot best stats
        x = stats["min_area_hc"][index_best_score]
        y = stats["score"][index_best_score]
        axarr[1].axvline(x, 0, 1, linestyle="dashed", color="red", linewidth=0.5)
        axarr[1].plot(x, y, markersize=5, marker="o", color="red", label="best score")
        axarr[1].set_title(f"Stats Plot\nbest score = {y:.2E}")
        axarr[1].legend()
        plt.tight_layout()
    return fig


def plot_roc(stats):
    with plt.style.context("seaborn-darkgrid"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        ax.plot(stats["FPR"], stats["TPR"])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("ROC Curve")
    return fig


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Estimate best minimum area & classification threshold pair for High Contrast Anomaly Detection Pipeline.",
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
        help="view dataset to finetune on",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=False,
        metavar="",
        choices=["ssim", "l1", "l2", "combined", "hsv"],
        default="l1",
        help="method used to compute resmaps",
    )

    # parse arguments
    args = parser.parse_args()

    # run main function
    main(model_path=args.path, view=args.view, method=args.method)

# python3 finetune.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 -v a00 -m l2
