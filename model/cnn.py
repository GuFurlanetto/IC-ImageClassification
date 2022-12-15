from tensorflow.keras.applications import mobilenet, resnet50, vgg16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, \
    Input, MaxPooling2D, Flatten, BatchNormalization, GlobalMaxPooling2D, Activation, \
    Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import tensorflow.keras as tk
import tensorflow_addons as tfa
import tensorflow as tf
from datetime import timedelta, timezone, datetime
from model.config import save_config, load_config
import numpy as np
import cv2
import os
import sys
import mlflow

class CNN:
    """
    Main neural network class.
    """

    def __init__(self, config):
        self.last_weight = None
        self.model = None
        self.config = config
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()

    def build(self, input_size):
        """
        Construct the model structure and print a summary

        :param input_size: Input size of the model
        """
        self.config.SIZE = input_size
        print(input_size)
        # Constructing a simple sequential model

        if self.config.ARCHITECTURE == 'cnn':  # Basic CNN
            self.model = Sequential()
            self.model.add(Input(shape=(input_size[1], input_size[0], input_size[2])))
            
            self.model.add(Conv2D(16, (3, 3), kernel_initializer='he_uniform', kernel_regularizer='l1'))
            self.model.add(Conv2D(16, (3, 3), kernel_initializer='he_uniform', kernel_regularizer='l1'))
            self.model.add(MaxPooling2D((3, 3)))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

            self.model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', kernel_regularizer='l1'))
            self.model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', kernel_regularizer='l1'))
            self.model.add(MaxPooling2D((3, 3)))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

        elif self.config.ARCHITECTURE == 'resnet50':  # Second tested model RESNET50
            new_input = Input(shape=(input_size[1], input_size[0], input_size[2]))
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
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

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
        last_log_dir = log_dir + self.config.RUN_NAME
        os.mkdir(last_log_dir)

        # Weight path to be used in detection stage
        self.last_weight = last_log_dir + '/'

        # Save config used for training and create a Checkpoint and plot callbacks
        save_config(self.config, 'Config', last_log_dir + '/')
        # cp_callback = ModelCheckpoint(filepath=last_log_dir + '/',
        #                               save_weights_only=True,
        #                               verbose=1)

        # Compile the model with hyper-parameters given in the Config
        self.model.compile(optimizer=tk.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                           loss=self.config.LOSS,
                           metrics=[#tfa.metrics.F1Score(num_classes=2, threshold=self.config.THRESHOLD),
                                    tf.keras.metrics.Recall(thresholds=self.config.THRESHOLD),
                                    tf.keras.metrics.Precision(thresholds=self.config.THRESHOLD)])

        # If XML file is provided, extract cropped spots and y value for Training and validation
        # Else utilize provided data already cropped
        if xml_dir is not None:
            print("Cropping train images ...")
            data_train = tf.data.Dataset.from_generator(training_dataset.create_cropped_list,
                                                        output_signature=(tf.TensorSpec(shape=(1, self.config.SIZE[1],
                                                                                               self.config.SIZE[0],
                                                                                               self.config.SIZE[2]),
                                                                                        dtype=tf.float32),
                                                                          tf.TensorSpec(shape=(1,), dtype=tf.int64)))
            # Conta a quantidade de exemplos no dataset de treino
            # Contagem atual: 231646
            # count = 0
            # for tuple in data_train:
            #     count += 1
            
            # print(f"Train {count}")

            data_train.batch(128, drop_remainder=True).repeat()

            print("Train images cropped\n")

            print("Cropping validation images ...")
            data_val = tf.data.Dataset.from_generator(val_dataset.create_cropped_list,
                                                        output_signature=(tf.TensorSpec(shape=(1, self.config.SIZE[1],
                                                                                               self.config.SIZE[0],
                                                                                               self.config.SIZE[2]),
                                                                                        dtype=tf.float32),
                                                                          tf.TensorSpec(shape=(1,), dtype=tf.int64)))
            # Conta a quantidade de exemplos no dataset de validação
            # Contagem atual: 27720
            # count = 0
            # for tuple in data_val:
            #     count += 1
            
            # print(f"Val {count}")
            data_val.batch(128, drop_remainder=True).repeat()
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
        mlflow.set_experiment(experiment_name=self.config.EXPERIMENT_NAME)
        mlflow.start_run(run_name=self.config.RUN_NAME)
        return self.model.fit(data_train, epochs=self.config.EPOCHS,
                              validation_data=(data_val),
                              steps_per_epoch=231646//128, validation_steps=27720//128, batch_size=128
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
            cropped_image = predicts_data.create_cropped_list(True)
        else:
            cropped_image = predicts_data

        # Load weights on path if provided
        # Else use last weight path
        if weight_path:
            self.model.load_weights(weight_path)
        # else:
        #     assert self.last_weight is not None
        #     self.model.load_weights(self.last_weight)

        # Make predictions
        predicts = self.model.predict(cropped_image, verbose=1)
        for i in range(len(predicts)):
            if predicts[i] > self.config.THRESHOLD:
                predicts[i] = 1
            else:
                predicts[i] = 0
        if 'y_true' in locals():
            return predicts, y_true, self.last_weight
        else:
            return predicts, self.last_weight
