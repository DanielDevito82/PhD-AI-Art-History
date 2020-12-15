# -*- coding: utf-8 -*-
"""MyModel"""

# standard library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# external
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

class MyModel(BaseModel):
    """My Model Class"""
    def __init__(self, config):
        super().__init__(config)
        self.model = Sequential()
        self.output_channels = self.config.model.output

        self.dataset = None
        self.info = None
        self.batch_size = self.config.train.batch_size
        #self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches
        #self.val_subsplits = self.config.train.val_subsplits
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0

        self.image_size = self.config.data.image_size
        self.train_dataset = None
        self.test_dataset = None

    def load_data(self):
        """Loads and Preprocess data """
        self.train_dataset, self.test_dataset, columns = DataLoader().load_data(self.config)

    def build(self):
        """ Builds the Keras model based """
        self.model.add(
            Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.config.model.input, strides=(2, 2), padding='valid', activation='relu'))
        self.model.add(
             Conv2D(filters=32, kernel_size=(3, 3), padding="valid", input_shape=self.config.model.input, activation='relu'))
        self.model.add(
            Conv2D(filters=64, kernel_size=(3, 3), input_shape=self.config.model.input, padding="same",
                   activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

        self.model.add(
              Conv2D(filters=80, kernel_size=(1, 1), input_shape=self.config.model.input, padding="valid", activation='relu'))
        self.model.add(
               Conv2D(filters=192, kernel_size=(3, 3), input_shape=self.config.model.input, padding="valid", activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        self.model.add(Flatten())

        # self.model.add(Dense(256))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.3))

        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(5))
        self.model.add(Activation('sigmoid'))

    def train(self):
        """Compiles and trains the model"""
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=self.config.train.metrics)
        self.model.summary()
        early_stop = EarlyStopping(monitor=self.config.train.EarlyStopping.monitor,
                                   mode=self.config.train.EarlyStopping.mode,
                                   verbose=1,
                                   patience=self.config.train.EarlyStopping.patience)

        model_history = self.model.fit(x=self.train_dataset,
                                       epochs=self.epoches,
                                       callbacks=[early_stop],
                                       validation_data=self.test_dataset,
                                       )

        return model_history.history

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        predictions = []
        for image, mask in self.dataset.take(1):
            predictions.append(self.model.predict(image))

        return predictions


    # def distributed_train(self):
    #     mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    #     with mirrored_strategy.scope():
    #         self.model = tf.keras.Model(inputs=inputs, outputs=x)
    #         self.model.compile(...)
    #         self.model.fit(...)
    #
    #
    #     os.environ["TF_CONFIG"] = json.dumps(
    #         {
    #             "cluster":{
    #                 "worker": ["host1:port", "host2:port", "host3:port"]
    #             },
    #             "task":{
    #                  "type": "worker",
    #                  "index": 1
    #             }
    #         }
    #     )
    #
    #     multi_worker_mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    #     with multi_worker_mirrored_strategy.scope():
    #         self.model = tf.keras.Model(inputs=inputs, outputs=x)
    #         self.model.compile(...)
    #         self.model.fit(...)
    #
    #     parameter_server_strategy = tf.distribute.experimental.ParameterServerStrategy()
    #
    #     os.environ["TF_CONFIG"] = json.dumps(
    #         {
    #             "cluster": {
    #                 "worker": ["host1:port", "host2:port", "host3:port"],
    #                 "ps":  ["host4:port", "host5:port"]
    #             },
    #             "task": {
    #                 "type": "worker",
    #                 "index": 1
    #             }
    #         }
    #     )