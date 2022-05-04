from model.config import Config
from model.utils import training_transfer, get_metrics, save_in_txt, Dataset
from model.cnn import CNN
import numpy as np

dataset_dir_train = '/home/gustavo/Documentos/IC/SegmentedModel/dataset'
xml_train = '/home/gustavo/Documentos/IC/SegmentedModel/dataset/train'
val_xml = '/home/gustavo/Documentos/IC/SegmentedModel/dataset/val'
xml_file = '/home/gustavo/Documentos/IC/SegmentedModel/2013-02-22_06_05_00.xml'
log_dir = '/home/gustavo/Documentos/IC/SegmentedModel/logs/'


print("Loading train data ...")
train_data = Dataset(xml_file)
train_data.load_custom(dataset_dir_train, "train")
print("Train data loaded\n")

print("Loading val data ...")
val_data = Dataset(xml_file)
val_data.load_custom(dataset_dir_train, "val")
print("Val data loaded\n")

my_config = Config()
my_config.LEARNING_RATE = 0.01
my_config.EPOCHS = 10
my_config.ARCHITECTURE = 'cnn'

print("Building model ...")
my_model = CNN(my_config)
my_model.build(train_data.min_size)
print("Model ready for training\n")

print("Training model ...")
my_model.train(train_data, val_data, log_dir, xml_train, val_xml)
print("Model trained\n")
