import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from model.utils import show_image, get_points_from_xml, get_images_spaces
import os

folder = '/home/gustavo/Documentos/IC/SegmentedModel/images'
xml_file = '/home/gustavo/Documentos/IC/SegmentedModel/2013-02-22_06_05_00.xml'
images = []

for filename in os.listdir(folder):
    img = os.path.join(folder, filename)
    images.append(cv.imread(img))


