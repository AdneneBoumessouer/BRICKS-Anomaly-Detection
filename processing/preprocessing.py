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
        self.mask_dir = os.path.join(input_directory, "ground_truth", "masks")
        self.rescale = rescale  # TODO remove this attr
        self.shape = shape
        self.color_mode = color_mode

    def get_train_generator(self, batch_size, shuffle=True):
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(**config.train_datagen_args)

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

    def get_val_generator(self, batch_size, shuffle=True):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size with shuffle False.
        """
        validation_datagen = ImageDataGenerator(**config.val_test_datagen_args)

        # Generate validation batches with datagen.flow_from_directory()
        validation_generator = validation_datagen.flow_from_directory(
            directory=self.val_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return validation_generator

    def get_test_generator(self):
        """
        For test, pass nb_test_images as batch size with shuffle False.
        """
        # for test dataset, only rescaling
        test_datagen = ImageDataGenerator(**config.val_test_datagen_args)

        # get total number of test images
        nb_test_images = self.get_total_number_test_images()

        # generate test batches with datagen.flow_from_directory()
        test_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=nb_test_images,
            class_mode="input",
            shuffle=False,
        )
        return test_generator

    def get_mask_generator(self):
        # for test dataset, only rescaling
        test_datagen = ImageDataGenerator(**config.val_test_datagen_args)

        # get total number of test images
        nb_test_images = self.get_total_number_test_images()

        # generate mask batches with datagen.flow_from_directory()
        mask_generator = test_datagen.flow_from_directory(
            directory=self.mask_dir,
            target_size=self.shape,
            color_mode="grayscale",
            batch_size=nb_test_images,
            class_mode="input",
            shuffle=False,
        )
        return mask_generator

    def get_finetuning_generator(self, batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        datagen = ImageDataGenerator(**config.val_test_datagen_args)

        # Generate validation batches with datagen.flow_from_directory()
        finetuning_generator = datagen.flow_from_directory(
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

    def get_total_number_val_images(self):
        return len(list_imgs(self.val_data_dir))
