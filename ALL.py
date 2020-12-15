from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from configs import config
from utils.config import Config
from configs.config import CFG
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# standard library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

data_config = Config.from_json(CFG)
# read the labels
dataframe = pd.read_csv(data_config.data.path_labels_csv, sep=data_config.data.csv_sep)

# read the dataframe columns
columns = []
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
model = Sequential()
model.add(
    Conv2D(filters=96, kernel_size=(11, 11), input_shape=data_config.model.input, strides=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(optimizer=data_config.train.optimizer.type,
                   loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=data_config.train.metrics)
model.summary()

model_history = model.fit_generator(generator=generated_training_batches,
                                         epochs=1,
                                         steps_per_epoch=1,
                                         validation_data=generated_test_batches
                                         )