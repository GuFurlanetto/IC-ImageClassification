import argparse
import os
import glob
import cv2
from model.cnn import CNN
from model.config import load_config, Config
from model.utils import Dataset, show_image, get_metrics, save_in_txt


def main(args):
    print("Loading model ...")
    config = load_config(os.path.basename(args.config), args.config.removesuffix(f"/{os.path.basename(args.config)}"))
    model = CNN(config)
    print("Model loaded.\n\n")
    
    print("Loading predicts data ...")
    xml_file = "2013-02-22_06_05_00.xml"
    predict_data = Dataset(xml_file)
    predict_data.load_custom(args.dataset, "test")
    print("Predicts data loaded\n")

    print("Running inference ...")
    model.build(predict_data.min_size)
    predicts, _ = model.detect(predict_data, args.dataset + "/test/", args.weights)
    y_true = predict_data.return_y_true()
    metrics = get_metrics(predicts, y_true, predict_data, predict_data.dataset_dir)
    images_draw = predict_data.draw_all_rectangles(predicts)
    print("Inference complete")

    os.makedirs("inference_results", exist_ok=True)
    output_dir = "inference_results"
    save_in_txt(metrics, output_dir + "/")

    for i, image in enumerate(images_draw):
        cv2.imwrite(f"{output_dir}/{i}.jpg", image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script")

    parser.add_argument('-c', '--config', help="Config path", required=True)
    parser.add_argument('-d', '--dataset', help="Dataset path", required=True)
    parser.add_argument('-w', '--weights', help="Weights path", required=True)

    args = parser.parse_args()

    # # Debug args
    # print(f"Config arg: {args.config}")
    # print(f"Dataset arg: {args.dataset}")
    # print(f"Weights arg: {args.weights}")

    main(args)