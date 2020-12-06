import os
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from processing import preprocessing
from processing import resmaps
from processing import detection
from processing import utils
from processing import anomaly
from sklearn.metrics import confusion_matrix
from processing.utils import printProgressBar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    model_path, view, method, min_area_lc, min_area_hc, threshold_hc, localize=False
):
    # load model and info
    model, info, _ = utils.load_model_HDF5(model_path)
    input_dir = info["data"]["input_directory"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    vmin = info["preprocessing"]["vmin"]
    vmax = info["preprocessing"]["vmax"]

    # initialize preprocessor
    preprocessor = preprocessing.Preprocessor(
        input_directory=input_dir, rescale=rescale, shape=shape, color_mode=color_mode,
    )

    # get test generator
    nb_test_images = preprocessor.get_total_number_test_images()
    test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )

    # retrieve test images
    index_array, filenames_test = utils.get_indices(test_generator, view)
    categories = [filename.split("/")[0] for filename in filenames_test]
    imgs_test_input = test_generator._get_batches_of_transformed_samples(index_array)[0]

    # reconstruct validation inspection images (i.e predict)
    imgs_test_pred = model.predict(imgs_test_input)

    # calculate resmaps
    RC_test = resmaps.ResmapCalculator(
        imgs_input=imgs_test_input,
        imgs_pred=imgs_test_pred,
        color_out="grayscale",
        method=method,
        filenames=filenames_test,
        vmin=vmin,
        vmax=vmax,
    )
    resmaps_test = RC_test.get_resmaps()

    # instantiate detectors
    detector_lc = detection.LowContrastAnomalyDetector(vmax=0.2)
    detector_hc = detection.HighContrastAnomalyDetector(vmin=0.2)

    # set estimated detector parameters from validation
    detector_lc.set_min_area(min_area_lc)
    detector_hc.set_min_area(min_area_hc)
    detector_hc.set_threshold(threshold_hc)

    # predict
    y_pred_lc, anomaly_maps_lc = detector_lc.predict(resmaps_test)
    y_pred_hc, anomaly_maps_hc = detector_hc.predict(resmaps_test)
    y_pred = list(np.array(y_pred_lc) | np.array(y_pred_hc))

    # retrieve ground truth
    y_true = [0 if "good" in filename.split("/") else 1 for filename in filenames_test]

    # confusion matrix
    tnr, _, _, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

    # create directory to save test results and classification stats
    save_dir = os.path.join(os.path.dirname(model_path), "test", method, view)
    os.makedirs(save_dir, exist_ok=True)

    # save test params
    test_params = {
        "method": method,
        "LowContrastAnomalyDetector": {
            "min_area": min_area_lc,
            # "threshold": threshold_lc,
        },
        "HighContrastAnomalyDetector": {
            "min_area": min_area_hc,
            "threshold": threshold_hc,
        },
        # "TPR": float(tpr),
        # "TNR": float(tnr),
        # "score": 0.7 * float(tpr) + 0.3 * float(tnr),
    }
    with open(os.path.join(save_dir, "test_params.json"), "w") as json_file:
        json.dump(test_params, json_file, indent=4, sort_keys=False)

    # save classification of test images
    classification = {
        "category": categories,
        "filename": filenames_test,
        "y_pred_lc": y_pred_lc,
        "y_pred_hc": y_pred_hc,
        "y_pred": y_pred,
        "y_true": y_true,
        "accurate_prediction": np.array(y_true) == np.array(y_pred),
    }

    df_clf = pd.DataFrame.from_dict(classification)
    df_clf.to_pickle(os.path.join(save_dir, "df_clf.pkl"))
    with open(os.path.join(save_dir, "classification.txt"), "w") as f:
        f.write(df_clf.to_string(header=True, index=True))

    # get and save classification stats
    df_stats_cb = utils.get_stats(df_clf, detector_type="cb")
    df_stats_lc = utils.get_stats(df_clf, detector_type="lc")
    df_stats_hc = utils.get_stats(df_clf, detector_type="hc")

    os.makedirs(os.path.join(save_dir, "stats"), exist_ok=True)

    df_stats_cb.to_pickle(os.path.join(save_dir, "stats", "df_stats_cb.pkl"))
    df_stats_lc.to_pickle(os.path.join(save_dir, "stats", "df_stats_lc.pkl"))
    df_stats_hc.to_pickle(os.path.join(save_dir, "stats", "df_stats_hc.pkl"))

    with open(os.path.join(save_dir, "stats", "df_stats_cb.txt"), "w") as f:
        f.write(df_stats_cb.to_string(header=True, index=True))

    with open(os.path.join(save_dir, "stats", "df_stats_lc.txt"), "w") as f:
        f.write(df_stats_lc.to_string(header=True, index=True))

    with open(os.path.join(save_dir, "stats", "df_stats_hc.txt"), "w") as f:
        f.write(df_stats_hc.to_string(header=True, index=True))

    # print classification stats to console
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_stats_cb)

    # print test params to console
    logger.info("\ntest_params: {}\n\n".format(test_params))

    # generate and save anomaly figure
    if localize:
        n = len(imgs_test_input)
        printProgressBar(0, n, prefix="Progress:", suffix="Complete", length=80)
        for i in range(n):
            printProgressBar(
                iteration=i + 1,
                total=n,
                prefix="Progress:",
                suffix="Complete | {}".format(filenames_test[i]),
                length=80,
            )
            fig = anomaly.generate_anomaly_localization_figure(
                img_input=imgs_test_input[i],
                img_pred=imgs_test_pred[i],
                resmap=resmaps_test[i],
                anomap_lc=anomaly_maps_lc[i],
                anomap_hc=anomaly_maps_hc[i],
                filename=filenames_test[i],
            )
            fname = os.path.join(save_dir, "anomaly_localization", filenames_test[i])
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fig.savefig(fname)
            plt.close(fig=fig)


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Classify test images with Low & High Anomaly Detection Pipelines.",
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
    parser.add_argument(
        "-alc",
        "--min-area-lc",
        type=int,
        required=True,
        metavar="",
        help="minimum area parameter for Low Contrast Anomaly Detection Pipeline",
    )
    parser.add_argument(
        "-ahc",
        "--min-area-hc",
        type=int,
        required=True,
        metavar="",
        help="minimum area parameter for High Contrast Anomaly Detection Pipeline",
    )
    parser.add_argument(
        "-thc",
        "--threshold_hc",
        type=float,
        required=True,
        metavar="",
        help="classification threshold for High Contrast Anomaly Detection Pipeline",
    )
    parser.add_argument(
        "-loc",
        "--localize",
        action="store_true",
        help="whether to generate and save anomaly localization figures for test images",
    )

    args = parser.parse_args()

    main(
        model_path=args.path,
        view=args.view,
        method=args.method,
        min_area_lc=args.min_area_lc,
        min_area_hc=args.min_area_hc,
        threshold_hc=args.threshold_hc,
        localize=args.localize,
    )

# Examples of command to initiate testing
# python3 test.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 -v a00 -m l2 -alc 89 -ahc 50 -thc 0.2 -loc
# python3 test.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 -v a45 -m l2 -alc 67 -ahc 20 -thc 0.206 -loc
