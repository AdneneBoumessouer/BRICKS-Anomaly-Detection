import os
import argparse
from processing.preprocessing import Preprocessor
from processing import utils
from processing import postprocessing

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
    preprocessing_function = info["preprocessing"]["preprocessing"]

    preprocessor = Preprocessor(
        input_directory=input_dir,
        rescale=rescale,
        shape=shape,
        color_mode=color_mode,
        preprocessing_function=preprocessing_function,
    )

    # get inspection images' filenames
    (
        filenames_val_insp,
        filenames_test_insp,
    ) = utils.get_inspection_filenames_from_config(input_dir)

    # -------------- INSPECTING VALIDATION IMAGES --------------
    logger.info("generating inspection plots of validation images...")

    inspection_val_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    imgs_val_input = inspection_val_generator.next()[0]
    filenames_val = inspection_val_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)

    # instantiate ResmapPlotter object to compute validation resmaps
    postproc_val = postprocessing.ResmapPlotter(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        filenames=filenames_val,
        color="grayscale",
        vmin=vmin,
        vmax=vmax,
    )

    fig_res_val = postproc_val.generate_inspection_figure(
        filenames_val_insp, model_path
    )
    fig_res_val.savefig(os.path.join(save_dir, "fig_insp_val.svg"))

    # -------------- INSPECTING TEST IMAGES --------------
    logger.info("generating inspection plots of test images...")

    nb_test_images = preprocessor.get_total_number_test_images()

    inspection_test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )

    imgs_test_input = inspection_test_generator.next()[0]
    filenames_test = inspection_test_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_test_pred = model.predict(imgs_test_input)

    # instantiate ResmapPlotter object to compute test resmaps
    postproc_test = postprocessing.ResmapPlotter(
        imgs_input=imgs_test_input,
        imgs_pred=imgs_test_pred,
        filenames=filenames_test,
        color="grayscale",
        vmin=vmin,
        vmax=vmax,
    )

    fig_res_test = postproc_test.generate_inspection_figure(
        filenames_test_insp, model_path
    )
    fig_res_test.savefig(os.path.join(save_dir, "fig_insp_test.svg"))

    fig_score_insp = postproc_test.generate_score_scatter_plot(
        inspection_test_generator, model_path, filenames_test_insp
    )
    fig_score_insp.savefig(os.path.join(save_dir, "fig_score_insp.svg"))

    fig_score_test = postproc_test.generate_score_scatter_plot(
        inspection_test_generator, model_path
    )
    fig_score_test.savefig(os.path.join(save_dir, "fig_score_test.svg"))
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

