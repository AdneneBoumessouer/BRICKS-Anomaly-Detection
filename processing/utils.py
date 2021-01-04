import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
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
        model = tf.keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": tf.keras.layers.LeakyReLU,
                "loss": losses.mssim_loss(dynamic_range),
                "mssim": metrics.mssim_metric(dynamic_range),
            },
            compile=True,
        )

    elif loss == "ssim":
        model = tf.keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": tf.keras.layers.LeakyReLU,
                "loss": losses.ssim_loss(dynamic_range),
                "ssim": metrics.ssim_metric(dynamic_range),
            },
            compile=True,
        )

    else:
        model = tf.keras.models.load_model(
            filepath=model_path,
            custom_objects={
                "LeakyReLU": tf.keras.layers.LeakyReLU,
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


def get_indices(generator, view, categories=[]):
    if categories:
        index_arr = []
        filenames = []

        for category in categories:
            index_arr_cat = np.nonzero(
                generator.classes == generator.class_indices[category]
            )[0]
            filenames_cat = [generator.filenames[i] for i in index_arr_cat]
            filenames_cat_view = [
                filename
                for filename in filenames_cat
                if filename.split("/")[-1].split("_")[0] == view
            ]
            index_arr_cat_view = [
                index_arr_cat[i]
                for i, filename in enumerate(filenames_cat)
                if filename.split("/")[-1].split("_")[0] == view
            ]

            index_arr.extend(index_arr_cat_view)
            filenames.extend(filenames_cat_view)
    else:
        filenames = [
            filename
            for filename in generator.filenames
            if filename.split("/")[-1].split("_")[0] == view
        ]
        index_arr = [
            i
            for i, filename in enumerate(generator.filenames)
            if filename.split("/")[-1].split("_")[0] == view
        ]
    return index_arr, filenames


def printProgressBar(
    iteration,
    total,
    prefix="Progress",
    suffix="Complete",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
    verbose=1,
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
    if verbose:
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
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


def list_imgs(dirName):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [
            os.path.join(dirpath, filename)
            for filename in filenames
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]
    return listOfFiles


def get_stats(df_clf, detector="cb"):
    if detector == "cb":
        y_pred = "y_pred"
    elif detector == "lc":
        y_pred = "y_pred_lc"
    else:
        y_pred = "y_pred_hc"

    categories = sorted(list(set(df_clf["category"].values)))
    TPR = []
    TNR = []

    # stats per category
    for category in categories:
        df_clf_cat = df_clf[df_clf["category"] == category]
        accuracy = np.count_nonzero(df_clf_cat[y_pred] == df_clf_cat["y_true"]) / len(
            df_clf_cat
        )
        if category == "good":
            TNR.append(accuracy)
            TPR.append(None)
        else:
            TNR.append(None)
            TPR.append(accuracy)

    # stats overall
    categories.append("overall")
    df_clf_good = df_clf[df_clf["category"] == "good"]
    tnr_total = np.count_nonzero(df_clf_good[y_pred] == df_clf_good["y_true"]) / len(
        df_clf_good
    )
    TNR.append(tnr_total)
    df_clf_defect = df_clf[df_clf["category"] != "good"]
    tpr_total = np.count_nonzero(
        df_clf_defect[y_pred] == df_clf_defect["y_true"]
    ) / len(df_clf_defect)
    TPR.append(tpr_total)
    df_stats = pd.DataFrame.from_dict({"category": categories, "TPR": TPR, "TNR": TNR})
    return df_stats


def get_optimal_figsize(nrows, ncols, scale=2):
    w_a4, h_a4 = 8.3, 11.7
    l = min(h_a4 / nrows, w_a4 / ncols)
    figsize = (scale * ncols * l, scale * nrows * l)
    return figsize

