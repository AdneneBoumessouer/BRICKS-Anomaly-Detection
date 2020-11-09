import os
import argparse
import tensorflow as tf
from autoencoder.autoencoder import AutoEncoder
from processing.preprocessing import Preprocessor
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
    )
    train_generator = preprocessor.get_train_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )
    validation_generator = preprocessor.get_val_generator(
        batch_size=autoencoder.batch_size, shuffle=False, purpose="val",
    )

    # find best learning rates for training
    lr_opt = autoencoder.find_lr_opt(train_generator, validation_generator, lr_estimate)

    # train with optimal learning rate
    autoencoder.fit(lr_opt=lr_opt, epochs=epochs, policy=policy)

    # save model and configuration
    autoencoder.save()

    # inspect validation and test images for visual assessement
    if args.inspect:
        inspection.inspect_images(model_path=autoencoder.save_path)
    logger.info("done.")
    return


def check_arguments(architecture, color_mode, loss):
    if loss == "mssim" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "ssim" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return


def print_info():
    if tf.test.is_gpu_available():
        print("GPU was detected...")
    else:
        print("No GPU was detected. CNNs can be very slow without a GPU...")
    print("Tensorflow version: {} ...".format(tf.__version__))
    print("Keras version: {} ...".format(tf.keras.__version__))
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

# python3 train.py -d LEGO_light/SV -a anoCAE -b 8 -l mssim -c rgb -e 60 -r custom --inspect
# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l mssim -c rgb -e 60 -r custom --inspect
# python3 train.py -d LEGO_light/SV -a baselineCAE -b 8 -l mssim -c rgb -e 60 -r custom --inspect
# python3 train.py -d LEGO_light/SV -a inceptionCAE -b 8 -l mssim -c rgb -e 60 -r custom --inspect
# python3 train.py -d LEGO_light/SV -a resnetCAE -b 8 -l mssim -c rgb -e 60 -r custom --inspect
# python3 train.py -d LEGO_light/SV -a skipCAE -b 8 -l mssim -c rgb -e 60 -r custom --inspect

# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l l2 -c rgb -e 100 -r custom
# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l l2 -c rgb -e 100 -r ktrain
# python3 train.py -d LEGO_light/SV -a mvtecCAE -b 8 -l l2 -c rgb -e 100 -r ktrain -p 1cycle

