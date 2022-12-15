"""
Mask R-CNN
Base Configurations class.
"""

import tensorflow.keras as tk
import inspect
import pickle


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.
# TODO: Revisar todas as configurações, limpar o código e adapatar para o nosso modelo
class Config:
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    SIZE = None

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to val with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # val stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Number of classification classes
    NUM_CLASSES = 2  # Override in sub-classes

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    L1_TERM = 0.001
    L2_TERM = 0.001
    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    # Controls the width of the network. This is known as the width multiplies in the MobileNet
    # paper. if alpha < 1.0, proportionally decreases the number of filters in each layer.
    # if alpha > 1.0, proportionally increases the number of filter in each layer.
    ALPHA = 1.0

    DROPOUT = 0.001

    MOMENTUM = 0.9

    EXPERIMENT_NAME = "None"
    RUN_NAME = "None"

    # String(name of optimizer) or optimizer instance
    OPTIMIZER = tk.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Loss function used in the model. Can be typically a tf.keras.losses. but can be a tf.keras.losses.Loss instance
    LOSS = tk.losses.BinaryCrossentropy()

    # List of all metrics used to evaluate the model
    METRICS = [tk.metrics.BinaryAccuracy(),
               tk.metrics.FalseNegatives()]

    THRESHOLD = 0.9

    # Number of epochs to val the model
    EPOCHS = 20
    ARCHITECTURE = 'cnn'

    SHUFFLE = True

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def print_txt(self, path):

        f = open(path + "config.txt", "a")

        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if i[0] != "display" and i[0] != "print_txt":
                    string = i[0] + ": " + str(i[1]) + "\n"
                    f.write(string)

        f.close()

def save_config(obj, name, dir):
    """
    Save the config file to later retrieval

    :param obj: Config object to be saved
    :param name: Name of the saved file
    :param dir: Directory to save
    """
    with open(dir + '/' +  name, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()



def load_config(name, dir):
    """
    Load config class from folder
    :param name: Name of the object file
    :param dir: Directory of the object file
    :return obj: Config object containing the attributes saved
    """
    with open(dir + '/' + name, 'rb') as f:
        return pickle.load(f)
        f.close()
