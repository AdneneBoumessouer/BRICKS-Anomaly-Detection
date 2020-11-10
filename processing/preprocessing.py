import os
import numpy as np
import random
from skimage.filters import gaussian
from skimage.util import random_noise
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from processing.utils import list_imgs
import config


class Preprocessor:
    def __init__(self, input_directory, rescale, shape, color_mode):
        self.input_directory = input_directory
        self.train_data_dir = os.path.join(input_directory, "train")
        self.val_data_dir = os.path.join(input_directory, "val")
        self.test_data_dir = os.path.join(input_directory, "test")
        self.rescale = rescale
        self.shape = shape
        self.color_mode = color_mode

    def get_train_datagen(self):
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=config.ROT_ANGLE,
            width_shift_range=config.W_SHIFT_RANGE,
            height_shift_range=config.H_SHIFT_RANGE,
            brightness_range=config.BRIGHTNESS_RANGE,
            shear_range=0.0,
            zoom_range=config.ZOOM_RANGE,
            channel_shift_range=config.CHANNEL_SHIFT_RANGE,
            fill_mode="nearest",
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=self.rescale,
            # preprocessing_function=preprocessing_function,
            preprocessing_function=None,
            data_format="channels_last",
            validation_split=0.0,
            dtype="float32",
        )
        return datagen

    def get_train_generator(self, batch_size, shuffle=True):
        # This will do preprocessing and realtime data augmentation:
        train_datagen = self.get_train_datagen()

        # Generate training batches with datagen.flow_from_directory()
        train_generator = train_datagen.flow_from_directory(
            directory=self.train_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return train_generator

    def get_val_datagen(self):
        # For validation dataset, only rescaling
        datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=None,
        )
        return datagen

    def get_val_generator(self, batch_size, shuffle=True, purpose="val"):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size with shuffle False.
        """
        # @todo: remove batch_size parameter
        if purpose == "val":
            validation_datagen = self.get_val_datagen()
        elif purpose == "monitor_train":
            validation_datagen = self.get_train_datagen()

        # get number of validation images
        nb_val_images = len(list_imgs(self.val_data_dir))

        # Generate validation batches with datagen.flow_from_directory()
        validation_generator = validation_datagen.flow_from_directory(
            directory=self.val_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=nb_val_images,
            class_mode="input",
            shuffle=shuffle,
        )
        return validation_generator

    def get_test_datagen(self):
        # For test dataset, only rescaling
        datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=None,
        )
        return datagen

    def get_test_generator(self, batch_size, shuffle=False):
        """
        For test, pass nb_test_images as batch size with shuffle False.
        """
        # For test dataset, only rescaling
        test_datagen = self.get_test_datagen()

        # Generate validation batches with datagen.flow_from_directory()
        test_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return test_generator

    def get_finetuning_generator(self, batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=None,
        )

        # Generate validation batches with datagen.flow_from_directory()
        finetuning_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return finetuning_generator

    def get_total_number_test_images(self):
        total_number = 0
        sub_dir_names = os.listdir(self.test_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(self.test_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            number = len(filenames)
            total_number = total_number + number
        return total_number


def preprocessing_function(img):
    """Add random noise to an image"""
    # blur image a little
    blurred = gaussian(img, sigma=0.4, multichannel=True)
    # add noise
    noisy = random_noise(blurred, mode="gaussian", clip=True, mean=0, var=1e-4)
    return noisy

