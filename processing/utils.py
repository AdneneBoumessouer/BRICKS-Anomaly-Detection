import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from autoencoder import metrics
from autoencoder import losses

import config


def get_model_info(model_path):
    dir_name = os.path.dirname(model_path)
    with open(os.path.join(dir_name, "info.json"), "r") as read_file:
        info = json.load(read_file)
    return info


def load_model_HDF5(model_path):
    """
    Loads model (HDF5 format), training setup and training history.
    """

    # load parameters
    info = get_model_info(model_path)
    loss = info["model"]["loss"]
    dynamic_range = info["preprocessing"]["dynamic_range"]

    # load autoencoder
    if loss == "mssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": losses.mssim_loss(dynamic_range),
                "mssim": metrics.mssim_metric(dynamic_range),
            },
            compile=True,
        )

    elif loss == "ssim":
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss": losses.ssim_loss(dynamic_range),
                "ssim": metrics.ssim_metric(dynamic_range),
            },
            compile=True,
        )

    else:
        model = keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2_loss": losses.l2_loss,
                "ssim": losses.ssim_loss(dynamic_range),
                "mssim": metrics.mssim_metric(dynamic_range),
            },
            compile=True,
        )

    # load training history
    dir_name = os.path.dirname(model_path)
    history = pd.read_csv(os.path.join(dir_name, "history.csv"))

    return model, info, history


def save_np(arr, save_dir, filename):
    np.save(
        file=os.path.join(save_dir, filename), arr=arr, allow_pickle=True,
    )


def get_list_imgs_dir(dirName):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [
            os.path.join(dirpath, filename)
            for filename in filenames
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]
    return listOfFiles


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def is_rgb(imgs):
    return imgs.ndim == 4 and imgs.shape[-1] == 3


def generate_new_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new


def save_dataframe_as_text_file(df, save_dir, filename):
    with open(os.path.join(save_dir, filename), "w+") as f:
        f.write(df.to_string(header=True, index=True))
        print("[INFO] validation_results.txt saved at {}".format(save_dir))


def get_inspection_filenames_from_config(input_dir):
    dataset_name = input_dir.split("/")[-1]
    if dataset_name == "SV":
        return config.SV_FILENAMES_VAL_INSPECTION, config.SV_FILENAMES_TEST_INSPECTION
    elif dataset_name == "TV":
        return config.TV_FILENAMES_VAL_INSPECTION, config.TV_FILENAMES_TEST_INSPECTION

