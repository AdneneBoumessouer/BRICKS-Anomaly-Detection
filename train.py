"""
Created on Tue Dec 10 19:46:17 2019

@author: Adnene Boumessouer
"""

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from autoencoder.autoencoder import AutoEncoder
from processing.preprocessing import Preprocessor
from processing.utils import printProgressBar as printProgressBar
from processing import utils
from processing import postprocessing
import inspection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Valid combinations for input arguments for architecture, color_mode and loss:

                        +----------------+----------------+
                        |       Model Architecture        |
                        +----------------+----------------+
                        |  mvtecCAE      |   ResnetCAE    |
                        |  baselineCAE   |                |
                        |  inceptionCAE  |                |
========================+================+================+
        ||              |                |                |
        ||   grayscale  |    ssim, l2    |    ssim, l2    |
Color   ||              |                |                |
Mode    ----------------+----------------+----------------+
        ||              |                |                |
        ||      RGB     |    mssim, l2   |    mssim, l2   |
        ||              |                |                |
--------+---------------+----------------+----------------+
"""


def check_arguments(architecture, color_mode, loss):
    if loss == "mssim" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "ssim" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return


def main(args):

    # get parsed arguments from user
    input_dir = args.input_dir
    architecture = args.architecture
    color_mode = args.color
    loss = args.loss
    batch_size = args.batch
    epochs = args.epochs
    lr_estimate = args.lr_estimate
    policy = args.policy

    # check arguments
    check_arguments(architecture, color_mode, loss)

    # get autoencoder
    autoencoder = AutoEncoder(input_dir, architecture, color_mode, loss, batch_size)

    # load data as generators that yield batches of preprocessed images
    preprocessor = Preprocessor(
        input_directory=input_dir,
        rescale=autoencoder.rescale,
        shape=autoencoder.shape,
        color_mode=autoencoder.color_mode,
        preprocessing_function=autoencoder.preprocessing_function,
    )
    train_generator = preprocessor.get_train_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )
    validation_generator = preprocessor.get_val_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )

    # find best learning rates for training
    lr_opt = autoencoder.find_lr_opt(train_generator, validation_generator, lr_estimate)

    # train
    autoencoder.fit(lr_opt=lr_opt, epochs=epochs, policy=policy)

    # save model
    autoencoder.save()

    # if args.inspect:
    #     # get inspection images' filenames
    #     (
    #         filenames_val_insp,
    #         filenames_test_insp,
    #     ) = utils.get_inspection_filenames_from_config(input_dir)

    #     # -------------- INSPECTING VALIDATION IMAGES --------------
    #     logger.info("generating inspection plots of validation images...")

    #     # create a directory to save inspection plots
    #     inspection_val_dir = os.path.join(autoencoder.save_dir, "inspection_val")
    #     if not os.path.isdir(inspection_val_dir):
    #         os.makedirs(inspection_val_dir)

    #     inspection_val_generator = preprocessor.get_val_generator(
    #         batch_size=autoencoder.learner.val_data.samples, shuffle=False
    #     )

    #     imgs_val_input = inspection_val_generator.next()[0]
    #     filenames_val = inspection_val_generator.filenames

    #     # get reconstructed images (i.e predictions) on validation dataset
    #     logger.info("reconstructing validation images...")
    #     imgs_val_pred = autoencoder.model.predict(imgs_val_input)

    #     # instantiate TensorImages object to compute validation resmaps
    #     postproc_val = postprocessing.Postprocessor(
    #         imgs_input=imgs_val_input,
    #         imgs_pred=imgs_val_pred,
    #         filenames=filenames_val,
    #         color="grayscale",
    #         vmin=autoencoder.vmin,
    #         vmax=autoencoder.vmax,
    #     )

    #     fig_val = postproc_val.generate_inspection_figure(filenames_val_insp)
    #     fig_val.savefig(os.path.join(autoencoder.save_dir), "fig_insp_val.svg")

    #     # -------------- INSPECTING TEST IMAGES --------------
    #     logger.info("generating inspection plots of test images...")

    #     # create a directory to save inspection plots
    #     inspection_test_dir = os.path.join(autoencoder.save_dir, "inspection_test")
    #     if not os.path.isdir(inspection_test_dir):
    #         os.makedirs(inspection_test_dir)

    #     nb_test_images = preprocessor.get_total_number_test_images()

    #     inspection_test_generator = preprocessor.get_test_generator(
    #         batch_size=nb_test_images, shuffle=False
    #     )

    #     imgs_test_input = inspection_test_generator.next()[0]
    #     filenames_test = inspection_test_generator.filenames

    #     # get reconstructed images (i.e predictions) on validation dataset
    #     logger.info("reconstructing test images...")
    #     imgs_test_pred = autoencoder.model.predict(imgs_test_input)

    #     # instantiate TensorImages object to compute test resmaps
    #     postproc_test = postprocessing.Postprocessor(
    #         imgs_input=imgs_test_input,
    #         imgs_pred=imgs_test_pred,
    #         filenames=filenames_test,
    #         color="grayscale",
    #         vmin=autoencoder.vmin,
    #         vmax=autoencoder.vmax,
    #     )

    #     fig_test = postproc_test.generate_inspection_figure(filenames_test_insp)
    #     fig_test.savefig(os.path.join(autoencoder.save_dir), "fig_insp_test.svg")

    if args.inspect:
        inspection.main(model_path=autoencoder.save_dir)
    logger.info("done.")
    return


def print_info():
    if tf.test.is_gpu_available():
        print("GPU was detected...")
    else:
        print("No GPU was detected. CNNs can be very slow without a GPU...")
    print("Tensorflow version: {} ...".format(tf.__version__))
    print("Keras version: {} ...".format(keras.__version__))
    return


def create_parser():
    parser = argparse.ArgumentParser(
        description="Train an AutoEncoder on an image dataset.",
        epilog="Example usage: python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale",
    )

    parser.add_argument(
        "-d",
        "--input-dir",
        type=str,
        required=True,
        metavar="",
        help="directory containing training images",
    )

    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=False,
        metavar="",
        choices=[
            "anoCAE",
            "mvtecCAE",
            "baselineCAE",
            "inceptionCAE",
            "resnetCAE",
            "skipCAE",
        ],
        default="mvtec2",
        help="architecture of the model to use for training: 'mvtecCAE', 'baselineCAE', 'inceptionCAE', 'resnetCAE' or 'skipCAE'",
    )

    parser.add_argument(
        "-c",
        "--color",
        type=str,
        required=False,
        metavar="",
        choices=["rgb", "grayscale"],
        default="grayscale",
        help="color mode for preprocessing images before training: 'rgb' or 'grayscale'",
    )

    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        required=False,
        metavar="",
        choices=["mssim", "ssim", "l2"],
        default="ssim",
        help="loss function to use for training: 'mssim', 'ssim' or 'l2'",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        required=False,
        metavar="",
        default=8,
        help="batch size to use for training",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=False,
        metavar="",
        default=None,
        help="Number of epochs to train",
    )

    parser.add_argument(
        "-r",
        "--lr-estimate",
        type=str,
        required=False,
        metavar="",
        choices=["custom", "ktrain"],
        default="custom",
        help="method to find optimal learning rate",
    )

    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        required=False,
        metavar="",
        choices=["cyclic", "1cycle"],
        default="cyclic",
        help="learning rate policy to train with",
    )

    parser.add_argument(
        "-i",
        "--inspect",
        action="store_true",
        help="generate inspection plots after training",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()  # added
    args = parser.parse_args()
    print_info()

    # run main function
    main(args)

# Examples of commands to initiate training with mvtec architecture LEGO_light/SV

# python3 train.py -d LEGO_light/SV -a anoCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d LEGO_light/SV -a baselineCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d LEGO_light/SV -a inceptionCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d LEGO_light/SV -a resnetCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d LEGO_light/SV -a skipCAE -b 8 -l mssim -c rgb --inspect

# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l l2 -c rgb -e 100 -r custom
# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l l2 -c rgb -e 100 -r ktrain
# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l l2 -c rgb -e 100 -r ktrain -p 1cycle

