from model.config import Config
from model.utils import training_transfer, get_metrics, save_in_txt
import numpy as np

dataset_dir_train = '/home/gustavo/Documentos/IC/SegmentedModel/dataset/train/'
dataset_dir_val = '/home/gustavo/Documentos/IC/SegmentedModel/dataset/val/'
dataset_dir_pred = '/home/gustavo/Documentos/IC/SegmentedModel/dataset/'
xml_file = '/home/gustavo/Documentos/IC/SegmentedModel/2013-02-22_06_05_00.xml'
xml_path_pred = '/home/gustavo/Documentos/IC/SegmentedModel/dataset/predicts/'
log_dir = '/home/gustavo/Documentos/IC/SegmentedModel/logs/NewTest/'


my_config = Config()
my_config.LEARNING_RATE = 0.01
my_config.EPOCHS = 3
my_config.ARCHITECTURE = 'cnn'

results, true, last_log = training_transfer(my_config, log_dir, dataset_dir_train, dataset_dir_val,
                                            dataset_dir_pred, xml_file)

metrics = get_metrics(results, true, 40)
save_in_txt(metrics, last_log)
