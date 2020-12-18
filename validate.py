import os
import argparse
import json
from processing import preprocessing
from processing import detection
from processing import utils
from processing import resmaps
import config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_path, view, method, min_area_hc):
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
    preprocessor = preprocessing.Preprocessor(
        input_directory=input_dir, rescale=rescale, shape=shape, color_mode=color_mode,
    )

    val_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    # retrieve validation images for specified view
    index_array, filenames = utils.get_indices(val_generator, view)
    imgs_val_input = val_generator._get_batches_of_transformed_samples(index_array)[0]

    # reconstruct validation inspection images (i.e predict)
    imgs_val_pred = model.predict(imgs_val_input)

    # calculate resmaps
    RC_val = resmaps.ResmapCalculator(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        color_out="grayscale",
        method=method,
        filenames=filenames,
        vmin=vmin,
        vmax=vmax,
    )
    resmaps_val = RC_val.get_resmaps()

    # instantiate detectors
    detector_lc = detection.LowContrastAnomalyDetector(vmin=0.1, vmax=0.30)
    detector_hc = detection.HighContrastAnomalyDetector(vmin=0.30, vmax=1.0)

    # fit detectors
    min_area_lc = detector_lc.estimate_area(resmaps_val)
    threshold_hc = detector_hc.estimate_threshold(resmaps_val, min_area=min_area_hc)

    # save validation results
    validation_result = {
        "LowContrastAnomalyDetector": {"min_area_lc": min_area_lc},
        "HighContrastAnomalyDetector": {
            "min_area_hc": min_area_hc,
            "threshold_hc": threshold_hc,
        },
    }
    print(validation_result)

    # save validation result
    save_dir = os.path.join(os.path.dirname(model_path), "validation", method, view)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "result_" + view + ".json"), "w") as json_file:
        json.dump(validation_result, json_file, indent=4, sort_keys=False)
    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Estimate best classification threshold for High Contrast Anomaly Detection Pipeline given a minimum area.",
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
        "-ahc",
        "--min-area-hc",
        type=int,
        required=True,
        metavar="",
        help="minimum area parameter for High Contrast Anomaly Detection Pipeline",
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
    main(
        model_path=args.path,
        view=args.view,
        method=args.method,
        min_area_hc=args.min_area_hc,
    )

# python3 validate.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 -v a00 -ahc 50 -m l2
# python3 validate.py -p saved_models/test_local_2/inceptionCAE_b8_e119.hdf5 -v a45 -ahc 20 -m l2
