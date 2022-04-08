from model.utils import Dataset
from model.cnn import CNN
from model.config import Config

val_dir = "/home/gustavo/Documentos/IC/imagens_dataset/Val"
train_dir = "/home/gustavo/Documentos/IC/imagens_dataset/Train"
xml_path = "/home/gustavo/Documentos/IC/SegmentedModel/2013-02-22_06_05_00.xml"
log_dir = "/home/gustavo/Documentos/IC/SegmentedModel/logs/"

print("Loading train dataset...")
train_set = Dataset(xml_path)
train_set.load_custom(train_dir)
print("Train dataset loaded\n")

print("Loading val dataset...")
val_set = Dataset(xml_path)
val_set.load_custom(val_dir)
print("Val dataset loaded\n")


class MyConfig (Config):
    EPOCHS = 10
    ARCHITECTURE = "cnn"


config = MyConfig()


my_model = CNN(config)
my_model.build(train_set.min_size)
my_model.train(train_set, val_set, log_dir, train_dir, val_dir)
my_model.detect()


