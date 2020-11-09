import os
import argparse
from processing.preprocessing import Preprocessor
from processing import utils
from processing import postprocessing
import config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_images(model_path):
    # load model for inspection
    logger.info("loading model for inspection...")
    model, info, _ = utils.load_model_HDF5(model_path)
    save_dir = os.path.dirname(model_path)

    input_dir = info["data"]["input_directory"]
    # architecture = info["model"]["architecture"]
    # loss = info["model"]["loss"]
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
    logger.info("generating inspection plots for validation images...")

    inspection_val_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    imgs_val_input = inspection_val_generator.next()[0]
    filenames_val = inspection_val_generator.filenames

    # get indices of validation inspection images
    val_insp_i = [
        filenames_val.index(filename) for filename in config.FILENAMES_VAL_INSPECTION
    ]
    imgs_val_input = imgs_val_input[val_insp_i]

    # reconstruct validation inspection images (i.e predict)
    imgs_val_pred = model.predict(imgs_val_input)

    # instantiate ResmapPlotter object to compute resmaps
    postproc_val = postprocessing.ResmapPlotter(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        filenames=config.FILENAMES_VAL_INSPECTION,
        color="grayscale",
        vmin=vmin,
        vmax=vmax,
    )

    # generate resmaps and save
    fig_res_val = postproc_val.generate_inspection_figure()
    fig_res_val.savefig(os.path.join(save_dir, "fig_insp_val.svg"))

    # -------------- INSPECTING TEST IMAGES --------------
    logger.info("generating inspection plots for test images...")

    nb_test_images = preprocessor.get_total_number_test_images()

    inspection_test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )
    # get preprocessed test images
    imgs_test_input = inspection_test_generator.next()[0]
    filenames_test = inspection_test_generator.filenames

    # get indices of test inspection images
    test_insp_i = [
        filenames_test.index(filename) for filename in config.FILENAMES_TEST_INSPECTION
    ]
    imgs_test_input = imgs_test_input[test_insp_i]

    # reconstruct inspection test images (i.e predict)
    imgs_test_pred = model.predict(imgs_test_input)

    # instantiate ResmapPlotter object to compute resmaps
    postproc_test = postprocessing.ResmapPlotter(
        imgs_input=imgs_test_input,
        imgs_pred=imgs_test_pred,
        filenames=config.FILENAMES_TEST_INSPECTION,
        color="grayscale",
        vmin=vmin,
        vmax=vmax,
    )

    # generate resmaps and save
    fig_res_test = postproc_test.generate_inspection_figure()
    fig_res_test.savefig(os.path.join(save_dir, "fig_insp_test.svg"))

    # --------------------------------------------------

    # fig_score_insp = postproc_test.generate_score_scatter_plot(
    #     inspection_test_generator, model_path, filenames_test_insp
    # )
    # fig_score_insp.savefig(os.path.join(save_dir, "fig_score_insp.svg"))

    # fig_score_test = postproc_test.generate_score_scatter_plot(
    #     inspection_test_generator, model_path
    # )
    # fig_score_test.savefig(os.path.join(save_dir, "fig_score_test.svg"))
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

