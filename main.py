from model.config import Config, save_config
from model.utils import get_metrics, save_in_txt, Dataset
from model.cnn import CNN
import mlflow

dataset_dir_train = '/home/gustavo/Documentos/IC/IC-ImageClassification/dataset'
xml_train = '/home/gustavo/Documentos/IC/IC-ImageClassification/dataset/train'
val_xml = '/home/gustavo/Documentos/IC/IC-ImageClassification/dataset/val'
pred_xml = '/home/gustavo/Documentos/IC/IC-ImageClassification/dataset/test'
xml_file = '/home/gustavo/Documentos/IC/IC-ImageClassification/2013-02-22_06_05_00.xml'
log_dir = '/home/gustavo/Documentos/IC/IC-ImageClassification/logs/'


print("Loading train data ...")
train_data = Dataset(xml_file)
train_data.load_custom(dataset_dir_train, "train")
print("Train data loaded\n")

print("Loading val data ...")
val_data = Dataset(xml_file)
val_data.load_custom(dataset_dir_train, "val")
print("Val data loaded\n")

print("Loading predicts data ...")
predict_data = Dataset(xml_file)
predict_data.load_custom(dataset_dir_train, "test")
print("Predicts data loaded\n")

my_config = Config()
my_config.LEARNING_RATE = 1e-4
my_config.EPOCHS = 3
my_config.THRESHOLD = 0.95
my_config.ARCHITECTURE = 'cnn'
my_config.EXPERIMENT_NAME = "TG Filter tests"
my_config.RUN_NAME = "019_filter8_split"

print("Building model ...")
my_model = CNN(my_config)
my_model.build(train_data.min_size)
print("Model ready for training\n")

print("Training model ...")
history = my_model.train(train_data, val_data, log_dir, xml_train, val_xml)
my_model.model.save_weights(my_model.last_weight + "/weights")
print("Model trained\n")
for i in range(my_config.EPOCHS):
    mlflow.log_metric("loss", history.history["loss"][i], step=i+1)
    mlflow.log_metric("precision", history.history["precision"][i], step=i+1)
    mlflow.log_metric("recall", history.history["recall"][i], step=i+1)
    mlflow.log_metric("val_loss", history.history["val_loss"][i], step=i+1)
    mlflow.log_metric("val precision", history.history["val_precision"][i], step=i+1)
    mlflow.log_metric("val recall", history.history["val_recall"][i], step=i+1)


print("Evaluating model ...")
results, _ = my_model.detect(predict_data, pred_xml)
y_true = predict_data.return_y_true()
metrics = get_metrics(results, y_true, predict_data, predict_data.dataset_dir)
save_in_txt(metrics, my_model.last_weight)
save_config(my_config, "Config.p", my_model.last_weight)

mlflow.log_metric("Test Accurary", metrics[0][0])
mlflow.log_metric("TP", metrics[2][0][0])
mlflow.log_metric("FP", metrics[2][0][1])
mlflow.log_metric("TN", metrics[2][0][2])
mlflow.log_metric("FN", metrics[2][0][3])
mlflow.log_metric("Test Precision", metrics[3][0])
mlflow.log_metric("Test Recall", metrics[4][0])
mlflow.log_metric("Test F1-Score", metrics[5][0])

mlflow.log_artifacts(my_model.last_weight)

mlflow.end_run()
print("Model evaluated")
