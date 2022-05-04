from tensorflow.keras.applications import mobilenet, resnet50, vgg16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, \
                                    Input, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from datetime import timedelta, timezone, datetime
from model.config import save_config, load_config
import numpy as np
import cv2
import os
import sys



class CNN:
    """
    Main neural network class.
    """

    def __init__(self, config):
        self.last_weight = None
        self.model = None
        self.config = config

    def build(self, input_size):
        """
        Construct the model structure and print a summary

        :param input_size: Input size of the model
        """
        self.config.SIZE = input_size
        # Constructing a simple sequential model
        if self.config.ARCHITECTURE == 'cnn':  # Basic CNN
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), activation='relu',
                                  input_shape=(input_size[0], input_size[1], input_size[2]),
                                  kernel_initializer='he_uniform'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D((2, 2)))

        elif self.config.ARCHITECTURE == 'resnet50':  # Second tested model RESNET50
            new_input = Input(shape=(input_size))
            resnet = resnet50.ResNet50(include_top=False, input_tensor=new_input, classes=2,
                                       weights='imagenet')

            for layer in resnet.layers:
                layer.trainable = False

            self.model = Sequential()
            self.model.add(resnet)

        elif self.config.ARCHITECTURE == 'mobilenet':  # Third Test MobileNet
            new_input = Input(shape=input_size)
            mobilenet_model = mobilenet.MobileNet(include_top=False, input_tensor=new_input, classes=2,
                                                  weights='imagenet')

            for layer in mobilenet_model.layers:
                layer.trainable = False

            self.model = Sequential()
            self.model.add(mobilenet_model)
        elif self.config.ARCHITECTURE == 'vgg16':  # Fourth Test VGG16
            new_input = Input(shape=input_size)
            vgg16_model = vgg16.VGG16(include_top=False, input_tensor=new_input, classes=2,
                                      weights='imagenet')

            for layer in vgg16_model.layers:
                layer.trainable = False

            self.model = Sequential()
            self.model.add(vgg16_model)

        else:
            sys.exit("Error: Architecture not found")

        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

        self.model.summary()

    def train(self, training_dataset, val_dataset, log_dir,
              xml_dir=None, xml_dir_val=None, weight_path=None):
        """
        Compile and train the model on the data provided

        :param training_dataset: Dataset used for training, using Dataset class from Utils
        :param val_dataset: Dataset used for validation, using Dataset class from Utils
        :param log_dir: Path to directory where logs/weights and configs will be saved
        :param xml_dir: Path to directory where XML file for each image in training is located
                        if none are given, assume already cropped images
        :param xml_dir_val: Path to directory where XML file for each image in validation is located
                            if none are given, assume already cropped images
        :param weight_path: Weight path to initialize, if None assume randomly created weights
        """

        # Prepare a folder to be used for log
        # Include '(last)' to identify last model trained
        for folder_name in os.listdir(log_dir):
            if "(last)" in folder_name:
                os.rename(os.path.join(log_dir, folder_name),
                          os.path.join(log_dir, folder_name.removesuffix("(last)")))

        # Load weight if needed
        if weight_path is not None:
            self.model.load_weights(weight_path)

        # Calculate time at the moment of training to use as a name for log folder
        difference = timedelta(hours=-3)
        time_zone = timezone(difference)
        data = datetime.now()
        last_log_dir = log_dir + data.astimezone(time_zone).strftime('%d%m%Y_%H:%M') + "(last)"
        os.mkdir(last_log_dir)

        # Weight path to be used in detection stage
        self.last_weight = last_log_dir + '/'

        # Save config used for training and create a Checkpoint and plot callbacks
        save_config(self.config, 'Config', last_log_dir + '/')
        cp_callback = ModelCheckpoint(filepath=last_log_dir + '/',
                                                         save_weights_only=True,
                                                         verbose=1)

        # Compile the model with hyper-parameters given in the Config
        self.model.compile(optimizer=self.config.OPTIMIZER, loss=self.config.LOSS, metrics=['accuracy'])

        # If XML file is provided, extract cropped spots and y value for Training and validation
        # Else utilize provided data already cropped
        if xml_dir is not None:
            print("Cropping train images ...\n")
            data_train = tf.data.Dataset.from_generator(training_dataset.create_cropped_list,
                                                        output_signature=(tf.TensorSpec(shape=(None, self.config.SIZE[0], self.config.SIZE[1], self.config.SIZE[2]),
                                                                                        dtype=tf.int32),
                                                                          tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
                                                                          )
                                                        ).repeat()
            print("Train images cropped\n")
            print("Cropping validation images ...\n")
            data_val = tf.data.Dataset.from_generator(val_dataset.create_cropped_list,
                                                      output_signature=(tf.TensorSpec(shape=(None, self.config.SIZE[0], self.config.SIZE[1], self.config.SIZE[2]),
                                                                                      dtype=tf.int32),
                                                                        tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
                                                                        )
                                                      ).repeat()
            print("Validation images cropped\n")
        else:
            cropped_image, y_true = np.array(training_dataset.images), np.array(training_dataset.y_true)
            cropped_image_val, y_true_val = np.array(val_dataset.images), np.array(val_dataset.y_true)
        # if xml_dir is not None:
        #     cropped_image, y_true = training_dataset.create_cropped_list(xml_dir)
        #     cropped_image_val, y_true_val = val_dataset.create_cropped_list(xml_dir_val)
        # else:
        #     cropped_image, y_true = np.array(training_dataset.images), np.array(training_dataset.y_true)
        #     cropped_image_val, y_true_val = np.array(val_dataset.images), np.array(val_dataset.y_true)

        # train the model using hyper-parameter in the Config
        return self.model.fit(data_train, epochs=self.config.EPOCHS,
                              validation_data=(data_val),
                              shuffle=self.config.SHUFFLE, callbacks=cp_callback,
                              steps_per_epoch=3460, validation_steps=1728
                              )

    def detect(self, predicts_data, xml_path=None, weight_path=None):
        """
        Make detection on images using the model

        :param predicts_data: list of images to be predict
        :param xml_path: Folder with XML files containing coordinates of each spot in each image
                         if none are provided, assume already cropped list
        :param weight_path: Weights to be used on the model, if none are given utilize folder '(last)'
        in log directory
        :return predicts: list containing a classification for each spot, 1 = occupied, 0 = empty
        """

        # Get all spots cropped from the image if a path is provided
        # Else assume images already cropped
        if xml_path is not None:
            cropped_image, y_true = predicts_data.create_cropped_list(xml_path)
        else:
            cropped_image = predicts_data

        # Load weights on path if provided
        # Else use last weight path
        if weight_path:
            self.model.load_weights(weight_path)
        else:
            assert self.last_weight is not None
            self.model.load_weights(self.last_weight)

        # Make predictions
        predicts = self.model.predict(cropped_image)
        predicts = np.argmax(predicts, axis=1)

        if 'y_true' in locals():
            return predicts, y_true, self.last_weight
        else:
            return predicts, self.last_weight

