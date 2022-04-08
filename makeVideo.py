import sys

from model.utils import make_video, natural_keys
import cv2
import os

folder = sys.argv[1]
imagens = []

for filename in sorted(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        imagens.append(img)

make_video(imagens, 5)
