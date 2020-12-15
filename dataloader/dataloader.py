# -*- coding: utf-8 -*-
"""Data Loader"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from configs import config
from utils.config import Config

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config, columns=[]):
        """
        Loads dataset from path
        Generates batches of images

        Args:
            data_config_ JSON File with configes for the below methods
        Returns:
            generated_training_batches: batches of images
        """
#        config=Config.from_json(data_config)
        # read the labels
        dataframe = pd.read_csv(data_config.data.path_labels_csv, sep=data_config.data.csv_sep)

        # read the dataframe columns

        for x in dataframe.columns:
            if x != 'image':
                columns.append(x)

        train_data_gen = ImageDataGenerator(rotation_range=0,  # rotate the image 30 degrees
                                            width_shift_range=0.0,  # Shift the pic width by a max of 10%
                                            height_shift_range=0.0,  # Shift the pic height by a max of 10%
                                            rescale=1 / 255,  # Rescale the image by normalzing it.
                                            shear_range=0.1,  # Shear means cutting away part of the image (max 20%)
                                            zoom_range=0.1,  # Zoom in by 20% max
                                            horizontal_flip=False,  # Allo horizontal flipping
                                            fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                                            )
        generated_training_batches = train_data_gen.flow_from_dataframe(
            dataframe=dataframe[:150],
            directory=data_config.data.folder,
            color_mode='grayscale',
            x_col="image",
            y_col=columns,
            batch_size=data_config.train.batch_size,
            seed=None,
            shuffle=True,
            class_mode="raw",
            target_size=(data_config.data.image_size, data_config.data.image_size)
        )

        # This functions helps if you are dealing with small datasets
        test_data_gen = ImageDataGenerator(rotation_range=0,  # rotate the image 30 degrees
                                           width_shift_range=0.0,  # Shift the pic width by a max of 10%
                                           height_shift_range=0.0,  # Shift the pic height by a max of 10%
                                           rescale=1 / 255,  # Rescale the image by normalzing it.
                                           shear_range=0.1,  # Shear means cutting away part of the image (max 20%)
                                           zoom_range=0.1,  # Zoom in by 20% max
                                           horizontal_flip=False,  # Allo horizontal flipping
                                           fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                                           )
        generated_test_batches = test_data_gen.flow_from_dataframe(
            dataframe=dataframe[151:],
            directory=data_config.data.folder,
            color_mode='grayscale',
            x_col="image",
            y_col=columns,
            batch_size=data_config.train.batch_size,
            seed=None,
            shuffle=True,
            class_mode="raw",
            target_size=(data_config.data.image_size, data_config.data.image_size)
        )

        return generated_training_batches, generated_test_batches, columns

