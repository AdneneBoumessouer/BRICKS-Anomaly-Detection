import os
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from processing.preprocessing import Preprocessor
from processing.resmaps import ResmapCalculator
from processing import utils
from processing.utils import printProgressBar
from skimage import measure
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_classes(resmaps_test, min_area, th):
    # 0 for defect-free, 1 for defective
    imgs_binary = resmaps_test > th
    imgs_labeled = np.array([measure.label(binary) for binary in imgs_binary])
    predictions = []
    for labeled in imgs_labeled:
        pred = 0
        for regionprop in measure.regionprops(labeled):
            if regionprop.area > min_area:
                pred = 1
                break
        predictions.append(pred)
    return predictions


# def save_segmented_images(resmaps, threshold, filenames, save_dir):
#     # threshold residual maps with the given threshold
#     resmaps_th = resmaps > threshold
#     # create directory to save segmented resmaps
#     seg_dir = os.path.join(save_dir, "segmentation")
#     if not os.path.isdir(seg_dir):
#         os.makedirs(seg_dir)
#     # save segmented resmaps
#     for i, resmap_th in enumerate(resmaps_th):
#         fname = utils.generate_new_name(filenames[i], suffix="seg")
#         fpath = os.path.join(seg_dir, fname)
#         plt.imsave(fpath, resmap_th, cmap="gray")
#     return


def main(model_path, view, method, min_area, th):
    # load model and info
    model, info, _ = utils.load_model_HDF5(model_path)
    input_dir = info["data"]["input_directory"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    vmin = info["preprocessing"]["vmin"]
    vmax = info["preprocessing"]["vmax"]

    # initialize preprocessor
    preprocessor = Preprocessor(
        input_directory=input_dir, rescale=rescale, shape=shape, color_mode=color_mode,
    )

    # get test generator
    nb_test_images = preprocessor.get_total_number_test_images()
    test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )

    # views = ["a00", "a45"]
    # for view, min_area, th in list(zip(views, min_areas, ths)):
    # retrieve test images
    filenames = [
        filename
        for filename in test_generator.filenames
        if filename.split("/")[-1].split("_")[0] == view
    ]
    index_array = [
        i
        for i, filename in enumerate(test_generator.filenames)
        if filename.split("/")[-1].split("_")[0] == view
    ]
    imgs_test_input = test_generator._get_batches_of_transformed_samples(index_array)[0]

    # reconstruct validation inspection images (i.e predict)
    imgs_test_pred = model.predict(imgs_test_input)

    # calculate resmaps
    RC_test = ResmapCalculator(
        imgs_input=imgs_test_input,
        imgs_pred=imgs_test_pred,
        color_out="grayscale",
        method=method,
        filenames=filenames,
        vmin=vmin,
        vmax=vmax,
        dtype="float64",
    )
    resmaps_test = RC_test.get_resmaps()

    # retrieve ground truth
    y_true = [0 if "good" in filename.split("/") else 1 for filename in filenames]

    # predict classes on test images
    y_pred = predict_classes(resmaps_test=resmaps_test, min_area=min_area, th=th)

    # confusion matrix
    tnr, fp, fn, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

    # initialize dictionary to store test results
    test_result = {
        "method": method,
        "min_area": min_area,
        "threshold": th,
        "TPR": tpr,
        "TNR": tnr,
        "score": (tpr + tnr) / 2,
    }

    # create directory to save test results
    save_dir = os.path.join(os.path.dirname(model_path), "test", view)
    if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    # save test result
    with open(os.path.join(save_dir, "result_" + view + ".json"), "w") as json_file:
        json.dump(test_result, json_file, indent=4, sort_keys=False)

    # save classification of image files in a .txt file
    classification = {
        "filenames": filenames,
        "predictions": y_pred,
        "truth": y_true,
        "accurate_predictions": np.array(y_true) == np.array(y_pred),
    }
    df_clf = pd.DataFrame.from_dict(classification)

    with open(os.path.join(save_dir, "classification_" + view + ".txt"), "w") as f:
        f.write(df_clf.to_string(header=True, index=True))

    # print classification results to console
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_clf)

    # print test_results to console
    logger.info(view + " test results: {}\n\n".format(test_result))


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
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
        "-a", "--area", type=int, required=True, metavar="", help="min_area"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=True,
        metavar="",
        help="classification threshold",
    )
    # parser.add_argument(
    #     "-s", "--save", action="store_true", help="save segmented images",
    # )

    args = parser.parse_args()

    main(args.path, args.view, args.method, args.area, args.threshold)

# Examples of command to initiate testing
# python3 test.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 --view a00 --method l1 --area 25 --threshold 0.5950
