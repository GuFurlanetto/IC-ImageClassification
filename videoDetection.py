"""
Main pipeline for real-time detection

:param: args: command line argument [Video file]
"""
import os.path
import time
import matplotlib.pyplot as plt
from model.utils import show_image, draw_rectangle, get_points_from_xml, \
                        get_image_similarity, save_comparison, make_video, cleanDirectory, \
                        save_in_txt, get_statistics, increase_brightness, equalizeColor

from model.cnn import CNN
from model.config import Config, load_config
import numpy as np
import cv2
import sys

p = time.process_time()
comparison_dir = '/home/gustavo/Documentos/IC/SegmentedModel/prev_spots/'
cleanDirectory(comparison_dir)

# Find the folder with weights from last training sessions
log_dir = sys.argv[3]

# Load config from last training session and build the model
custom_config = load_config('Config', log_dir)
print(custom_config.SIZE)
model = CNN(custom_config)
model.build(custom_config.SIZE)

# Get video file from command line argument adn extract points from XML file input
video_file = cv2.VideoCapture(sys.argv[1])
pt_A, pt_B, pt_C, pt_D, _ = get_points_from_xml(sys.argv[2])

i = 0
final_frame_list = []
# Run through each video frame and draw appropriate rectangle from status occupied or empty
t = time.process_time()
tempo = []
totaisFrames = 0
statistics = [[[], "Ocupação atual"], [[], "Média de ocupação"], [[], "Tempo médio de OCC"],
              [[], "Dia e hora de pico"]]
fps = 2

preTime = time.process_time() - p
while video_file.isOpened():
    ret, frame = video_file.read()
    f = time.process_time()

    if not ret:
        break

    # Calculate if the are any changes from spots capture in the last frame
    changes, index, images = get_image_similarity(frame, comparison_dir, sys.argv[2], custom_config.SIZE)
    # If no changes are detected, reuse predicts from last frame
    # Else redo classification in the spots that have benn detected changes
    if not changes:
        if 'predicts' in locals():
            marked_frame = draw_rectangle(frame, [pt_A, pt_B, pt_C, pt_D], predicts)
            final_frame_list.append(marked_frame)
        else:
            sys.exit("Err: Unexpected behaviour")
    else:
        # print("\nNova previsão para os seguintes spots:\n")
        # for i in index:
        #     print("ID:", i, "\n")
        # show_image("Mudou nesse frame", frame)

        spots = np.array([images[i][0] for i in index])
        predicts, _ = model.detect(spots, weight_path=log_dir + '/')

        for i in range(len(predicts)):
            images[index[i]][2] = "occupied" if predicts[i] == 1 else "empty"

        complete_predicts = [1 if images[i][2] == "occupied" else 0 for i in range(len(images))]

        marked_frame = draw_rectangle(frame, [pt_A, pt_B, pt_C, pt_D], complete_predicts)
        final_frame_list.append(marked_frame)

        print("Classificadas: ", len(predicts))
        print("Total: ", len(complete_predicts))
        print("\n")
        save_comparison(comparison_dir, images)
        # print("Frame processado em:", time.process_time() - f)
    tempo.append(time.process_time() - f)
    totaisFrames += 1

    get_statistics(statistics, complete_predicts, fps)

print("Tempo de build:", preTime)
print("Média de tempo por frame:", sum(tempo) / totaisFrames)
print("Tempo de processamento:", time.process_time() - t)
video_file.release()
make_video(final_frame_list, fps)
