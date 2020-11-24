import os
import argparse
from processing.preprocessing import Preprocessor
from processing import utils
from processing import postprocessing
import config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate(model_path):
    # load model for inspection
    logger.info("loading model for inspection...")
    model, info, _ = utils.load_model_HDF5(model_path)
    save_dir = os.path.dirname(model_path)

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

    # -------------- INSPECTING VALIDATION IMAGES --------------
    val_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    imgs_val_input = val_generator.next()[0]
    filenames_val = val_generator.filenames

    # retrieve validation images for front view a00
    a00_i = [i for i, filename in enumerate(
        filenames_val) if filename.split("/")[-1].split["_"][1] == "a00"]
    imgs_val_input_a00 = imgs_val_input[a00_i]
    # retrieve validation images for side view a45
    a45_i = [i for i, filename in enumerate(
        filenames_val) if filename.split("/")[-1].split["_"][1] == "a45"]
    imgs_val_input_a45 = imgs_val_input[a45_i]

    # reconstruct validation inspection images (i.e predict)
    imgs_val_pred_a00 = model.predict(imgs_val_input_a00)
    imgs_val_pred_a45 = model.predict(imgs_val_input_a45)

    # # instantiate ResmapPlotter object to compute resmaps
    # postproc_val = postprocessing.ResmapPlotter(
    #     imgs_input=imgs_val_input,
    #     imgs_pred=imgs_val_pred,
    #     filenames=config.FILENAMES_VAL_INSPECTION,
    #     color_out="grayscale",
    #     vmin=vmin,
    #     vmax=vmax,
    # )

    # # generate resmaps and save
    # fig_res_val = postproc_val.generate_inspection_figure()
    # fig_res_val.savefig(os.path.join(save_dir, "fig_insp_val.svg"))

    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Test model on some images for inspection.",
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )

    # parse arguments
    args = parser.parse_args()

    # run main function
    inspect_images(model_path=args.path)
