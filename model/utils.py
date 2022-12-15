import gc
import time
import xml.etree.ElementTree as elt
import matplotlib.pyplot as plt
from model.cnn import CNN
import numpy as np
import random
import sewar
import glob
import cv2
import os
import re
import gc
import itertools
import json


##################################################
# Regions extraction                             #
##################################################

def get_images_spaces(img, pt_A, pt_B, pt_C, pt_D):
    """
    Crop the images by all the spaces provided by the xml file
    :param pt_A, pt_B, pt_C, pt_D: points in the image where the image will be cropped
    :param img: Image to be cropped
    :return: List containing all the cropped images
    """

    cropped_images = []

    for i in range(len(pt_A)):
        # Here, I have used L2 norm.
        width_AD = np.sqrt(((pt_A[i][0] - pt_D[i][0]) ** 2) + ((pt_A[i][1] - pt_D[i][1]) ** 2))
        width_BC = np.sqrt(((pt_B[i][0] - pt_C[i][0]) ** 2) + ((pt_B[i][1] - pt_C[i][1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[i][0] - pt_B[i][0]) ** 2) + ((pt_A[i][1] - pt_B[i][1]) ** 2))
        height_CD = np.sqrt(((pt_C[i][0] - pt_D[i][0]) ** 2) + ((pt_C[i][1] - pt_D[i][1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A[i], pt_B[i], pt_C[i], pt_D[i]])
        output_pts = np.float32([[0, 0],
                                 [0, maxHeight - 1],
                                 [maxWidth - 1, maxHeight - 1],
                                 [maxWidth - 1, 0]])

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        cropped_images.append(cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR))

    return cropped_images


def get_points_from_xml(xml_file):
    """
    Get all coordinates from a XML file

    :param xml_file: Path to the XML file
    :return: 4 list containing [x, y] for each of the four coordinates plus a list of the y_true values
    """

    # Load the XML file
    tree = elt.parse(xml_file)
    root = tree.getroot()

    point_a = []
    point_b = []
    point_c = []
    point_d = []
    occupied_space = []

    # Find all spots and extracts their coordinates
    for space in root.findall("space"):
        if 'occupied' not in space.attrib:
            space.set("occupied", 0)
        point_a.append([int(space[1][0].attrib['x']), int(space[1][0].attrib['y'])])
        point_b.append([int(space[1][1].attrib['x']), int(space[1][1].attrib['y'])])
        point_c.append([int(space[1][2].attrib['x']), int(space[1][2].attrib['y'])])
        point_d.append([int(space[1][3].attrib['x']), int(space[1][3].attrib['y'])])
        occupied_space.append(int(space.attrib['occupied']))

    return point_a, point_b, point_c, point_d, occupied_space


def get_points_from_json(json_file, img_name, data_dir):
    with open(json_file, "r") as f:
        ann_file = json.load(f)

    for image in ann_file:
        if img_name in image['image']:
            found = True
            break
        else:
            found = False

    assert found == True
    
    img = cv2.imread(data_dir + "/" + os.path.basename(image['image'])[9:])
    pt_A = []
    pt_B = []
    pt_C = []
    pt_D = []
    y_true = []

    width, height = img.shape[:2] 
    for point in image['label']:
        pta, ptb, ptc, ptd = point['points']
        pt_A.append([int((pta[0] / 100) * height), int((pta[1] / 100) * width)])
        pt_B.append([int((ptb[0] / 100) * height), int((ptb[1] / 100) * width)])
        pt_C.append([int((ptc[0] / 100) * height), int((ptc[1] / 100) * width)])
        pt_D.append([int((ptd[0] / 100) * height), int((ptd[1] / 100) * width)])

        if point['polygonlabels'][0] == 'occupied':
            y_true.append(1)
        else:
            y_true.append(0)
    
    return pt_A, pt_B, pt_C, pt_D, y_true



#############################################################
# Image Visualization                                       #
#############################################################

def show_image(title, picture):
    """
    Display the image in a window

    :param title: tittle for the window
    :param picture: image to be display in cv2 format
    """

    cv2.imshow(title, picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_rectangle(image, coordinates, occupied):
    """
    Draw a rectangle in the given coordinates.
    Green for unoccupied and red for occupied

    :param image: Image where the rectangle will be draw
    :param coordinates: list of Rectangle coordinates to draw [pt_A, pt_B, pt_C, pt_D]
    :param occupied: list of Boolean value teling if the space is occupied
    """

    for i in range(len(occupied)):
        pts = np.array([[coordinates[0][i], coordinates[1][i],
                         coordinates[2][i], coordinates[3][i]]],
                       np.int32)

        pts = pts.reshape((-1, 1, 2))

        isClosed = True

        # Green for occupied
        # Red for unoccupied
        if not occupied[i]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.polylines() method
        # Draw a Blue polygon w
        # ith
        # thickness of 1 px
        image = cv2.polylines(image, [pts],
                              isClosed, color, thickness)

    return image


##########################################################
# Dataset                                                #
##########################################################


def image_generator(dataset_dir):
    temp = None

    # Load all images from de subset
    for filename in sorted(os.listdir(dataset_dir), reverse=True):
        if filename.endswith(".json"):
            continue
        if filename.endswith(".xml"):
            temp = os.path.join(dataset_dir, filename)
            continue
        else:
            img = cv2.imread(os.path.join(dataset_dir, filename))
            if "2015" in filename or "2016" in filename:
                temp = [os.path.basename(filename), os.path.join(dataset_dir, "result.json")]
            
            assert temp is not None
            yield img, temp


# def image_generator(dataset_dir):
#     # Load all images from de subset
#     for filename in sorted(os.listdir(dataset_dir), reverse=True):
#         img = cv2.imread(os.path.join(dataset_dir, filename))
#         if img is not None:
#             yield img


class Dataset:
    """
    Class responsible for process the dataset
    """

    def __init__(self, xml_path):
        self.min_size = None
        self.dataset_dir = None
        self.y_true = []
        self.images = None
        self.pt_A, self.pt_B, self.pt_C, self.pt_D, _ = get_points_from_xml(xml_path)

    def load_custom(self, dataset_dir, subset=None):
        """Load a subset of the image dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: val or train
        """

        # train, validation or predicts dataset?
        if subset is not None:
            assert subset in ["val", "train", "test"]
            dataset_dir = os.path.join(dataset_dir, subset)

        self.dataset_dir = dataset_dir
        self.images = image_generator(dataset_dir)
        self.get_min_size()

    def dataset_example(self, quantity):
        """
        Show a certain number of random images from the dataset

        :param quantity: Number of images to be displayed
        """

        # Assert that the images has been loaded
        assert len(self.images) != 0

        for number in np.random.randint(len(self.images), size=quantity):
            show_image("Example " + str(number), self.images[number])

    def get_min_size(self):
        """
        Calculate the min size of the dataset images
        """
        img = cv2.imread("2013-04-15_16_10_11.jpg")

        # Crop all parking lot spots on the image
        cropped_image = get_images_spaces(img, self.pt_A,
                                          self.pt_B, self.pt_C, self.pt_D)

        # Run through the data looking for the smaller image size
        self.min_size = cropped_image[0].shape
        for image in cropped_image:
            self.min_size = min(self.min_size, image.shape)
            
    def data_size(self):
        self.load_custom(self.dataset_dir)

        total_len = 0
        for _, xml_file in self.images:
            spots, _, _, _, _ = get_points_from_xml(xml_file)
            total_len += len(spots)

        return total_len

    def create_cropped_list(self, predict=False):
        """
        Create a cropped list containing all the spots in the dataset images

        :param predict:
        :param xml_path: Folder path containing all coordinates for each image spots
        :return cropped_images: Cropped spots from all images in the dataset
        :return y_true: y value for all cropped images, indicating occupation status
        """

        # Crop and resize all spots by the minimum size of the dataset
        for img, xml_file in self.images:
            if type(xml_file) == list:
                self.pt_A, self.pt_B, self.pt_C, self.pt_D, y = get_points_from_json(xml_file[1], xml_file[0], self.dataset_dir)
            else:
                self.pt_A, self.pt_B, self.pt_C, self.pt_D, y = get_points_from_xml(xml_file)

                
            crop = np.array(get_images_spaces(img, self.pt_A, self.pt_B,
                                              self.pt_C, self.pt_D))
            crop = resize_image(crop, self.min_size[0], self.min_size[1])

            # Normalize all pixels values
            crop = crop / 255.0

            # Transform y value to be accepted by the model
            # for i in range(len(y)):
            #     y[i] = np.array([0, 1]) if y[i] == 1 else np.array([1, 0])

            # y = np.array(y)

            for i in range(len(crop)):
                #show_image(str(y[i]), crop[i])
                crop_new_shape = np.expand_dims(crop[i], axis=0)
                y_new_shape = np.expand_dims(y[i], axis=0)

                if predict:
                    yield crop_new_shape
                else:
                    yield crop_new_shape, y_new_shape


    def return_y_true(self):
        y_final = []

        self.load_custom(self.dataset_dir)

        for img, xml_file in self.images:

            if type(xml_file) == list:
                _, _, _, _, y = get_points_from_json(xml_file[1], xml_file[0], self.dataset_dir)
            else:
                _, _, _, _, y = get_points_from_xml(xml_file)

            # Transform y value to be accepted by the model
            # for i in range(len(y)):
            #     y[i] = np.array([0, 1]) if y[i] == 1 else np.array([1, 0])
            #     y_final.append(y[i])

            y_final.extend(y)

        return y_final

    def draw_all_rectangles(self, predicts):
        """
        Draw all rectangles with the status predicts by the model and display all of them

        :param predicts: list of predicts generated by the model
        """
        frame_list = []
        last = 0
        self.load_custom(self.dataset_dir)

        # Go through all images drawing the rectangles
        for image, xml in self.images:

            if type(xml) == list:
                self.pt_A, self.pt_B, self.pt_C, self.pt_D, _ = get_points_from_json(xml[1], xml[0], self.dataset_dir)
            else:
                self.pt_A, self.pt_B, self.pt_C, self.pt_D, _ = get_points_from_xml(xml)

            predicts_in_use = predicts[last: last + len(self.pt_A)]
            last += len(self.pt_A)
            image_draw = draw_rectangle(image, [self.pt_A, self.pt_B,
                                                self.pt_C, self.pt_D],
                                        predicts_in_use)

            frame_list.append(image_draw)
            

        return frame_list

    def prepare_image_puc(self, dataset_dir, limits=None):
        """
        Prepare segmented images from de PUCPR dataset for training and get
        y value from all images in the dataset

        :param dataset_dir: Dataset directory
        """

        img_count = 0

        if limits is not None:
            lower_lim = limits[0]
            upper_limit = limits[1]
        else:
            lower_lim = 0
            upper_limit = 999999

        # Load all images from the dataset directory
        for folder in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, folder)
            for filename in os.listdir(path):
                img = cv2.imread(os.path.join(path, filename))
                if (img is not None) and (lower_lim < img_count < upper_limit):
                    ##### Test ######
                    img = equalizeColor(img)
                    #################
                    final_image = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    self.images.append(final_image)

                    if folder == 'occupied':
                        self.y_true.append(np.array([0, 1]))
                    else:
                        self.y_true.append(np.array([1, 0]))

                img_count += 1
                if img_count > upper_limit:
                    break

            img_count = 0

        # Resize
        self.get_min_size()
        resize_image(self.images, self.min_size[1], self.min_size[0])


##########################################################
# Image processing                                       #
##########################################################

##################### Testing area #########################
def equalizeColor(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


#########################################################

def make_video(frame_list, fps):
    height, width, _ = frame_list[0].shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    output = cv2.VideoWriter('ParkingLot.avi', fourcc, fps, size)

    for i in range(len(frame_list)):
        output.write(frame_list[i])

    output.release()


def resize_image(images, width, height):
    """
    Resize given image to a new size

    :param images: List of images to be resize
    :param width: Width of the new size
    :param height: Height of the new size
    :return resized_images: List containing all images with new sizes
    """

    cropped_resize = []

    for i in range(len(images)):
        tmp = cv2.resize(images[i], dsize=(width, height), interpolation=cv2.INTER_AREA)
        cropped_resize.append(tmp)

    return np.array(cropped_resize)

def night_filter(image):
    pass


##########################################################
# Similarity functions                                   #
##########################################################

def get_image_similarity(new_frame, comparison_dir, xml_file, size):
    """
    Tracks changes in all spots of the new frame in comparison with
    last detection.

    :param new_frame: Image to track similarity
    :param comparison_dir: Directory containing images of the last detection
    :param xml_file: xml file containing coordinates of all spots in the given image
    :return change: Boolean value representing if has been detected changes
    :return index: list containing index of the images tha have changes
    :return comparison_images: Images resulting the comparison
    """

    # Crop all spots in the new frame
    pt_a, pt_b, pt_c, pt_d, _ = get_points_from_xml(xml_file)
    cropped_spots = np.array(get_images_spaces(new_frame, pt_a, pt_b, pt_c, pt_d))
    resize_image(cropped_spots, size[1], size[0])

    # Normalize all pixels values
    cropped_spots = cropped_spots / 255.0

    # Load previous detections in the same spots
    # If directory is empty, than create a model list [nome, name, status]
    # containing all spots in the frame cropped and return for classification
    # Else load all comparison images into the list template [image, name, status]
    if len(os.listdir(comparison_dir + 'occupied')) < 1 and len(os.listdir(comparison_dir + 'empty')) < 1:
        comparison_images = []
        index = [i for i in range(len(cropped_spots))]
        for i in range(len(cropped_spots)):
            comparison_images.append([cropped_spots[i], "ID: " + str(i) + ".jpg", 0])

        return True, index, comparison_images
    else:
        comparison_images = load_comparison(comparison_dir)

    assert len(cropped_spots) == len(comparison_images)

    changes = False
    index = []
    # Calculate similarity between all new spots with the old ones
    # If the images are similar, save the new spots
    # If the images are not similar, ?
    for i in range(len(cropped_spots)):
        if sewar.uqi(cropped_spots[i] * 255.0, comparison_images[i][0]) < 0.85:
            changes = True
            index.append(i)

        comparison_images[i][0] = cropped_spots[i]

    # if no changes are detected save all images
    if not changes:
        save_comparison(comparison_dir, comparison_images)

    return changes, index, comparison_images


def load_comparison(comparison_dir):
    """
    Load all images saved for comparison

    :param comparison_dir: Comparisons directory
    :return comparison_list: List containing all images, names and status [Image, name, status]
    """
    comparison_list = []

    for folder in os.listdir(comparison_dir):
        files_folder = os.path.join(comparison_dir, folder)
        for filename in os.listdir(files_folder):
            file_path = os.path.join(files_folder, filename)
            comparison_list.append([cv2.imread(file_path), filename, folder])

    comparison_list.sort(key=lambda key: natural_keys(key[1]))

    return comparison_list


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    list.sort(key=natural_keys) sorts in human order
    """

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def save_comparison(comparison_dir, images_list):
    """
    Save all images in the comparison folder

    :param comparison_dir: Folder where the images will be saved
    :param images_list: List containing all images, names and status [Image, name]
    """

    cleanDirectory(comparison_dir)

    for image in images_list:
        folder = os.path.join(comparison_dir, image[2])
        file_name = os.path.join(folder, image[1])
        cv2.imwrite(file_name, image[0] * 255.0)


def cleanDirectory(directory):
    for folder in os.listdir(directory):
        folderPath = os.path.join(directory, folder)
        for file in os.listdir(folderPath):
            os.remove(os.path.join(folderPath, file))


#############################################################
# Transfer training                                         #
#############################################################

def training_transfer(config, log_dir, training_dir, val_dir, pred_dir, xml_file):
    print("Training the model ...")

    # Build model
    model = CNN(config)

    limit_pairs = [[5000, 15000], [15000, 25000], [25000, 35000], [35000, 45000]]

    val_data = Dataset(xml_file)
    train_data = Dataset(xml_file)

    val_data.prepare_image_puc(val_dir, [0, 10000])
    model.build(val_data.min_size)

    for limits in limit_pairs:
        print("\nInitiating partial training ...")

        train_data.prepare_image_puc(training_dir, limits)
        model.train(train_data, val_data, log_dir)
        del train_data.images[:]
        del train_data.y_true[:]

        print("Partial training complete")

    print("Training complete")
    del train_data
    del val_data
    gc.collect()

    print("Detecting ...")
    pred_data = Dataset(xml_file)
    pred_data.load_custom(pred_dir, 'predicts')

    predicts, log_dir, y_true = model.detect(pred_data, pred_dir + 'predicts/')
    print("Pipeline ended")

    return predicts, log_dir, y_true


######################################################
# Save config                                        #
######################################################

def save_in_txt(list_to_save, path):
    """
    Save the content of a list (2D maximum) in a txt file

    :param list_to_save: List with items to be saved (2D maximum) [[Item1, name], [Item2, name], ...]
    :param path: Directory path where the list will be saved
    """

    # Create the file and write all itens
    with open(path + 'metrics.txt', 'w') as f:
        for item in list_to_save:
            f.write(item[1] + ": ")

            if isinstance(item[0], list):
                for sub_item in item[0]:
                    f.write(str(sub_item) + ', ')
            else:
                f.write(str(item[0]))

            f.write("\n")


########################################################
# Statistics                                           #
########################################################

def get_metrics(predicts, y_true, test_data, data_dir):
    """
    Calculates and print performance metrics on predictions.
    Metrics to be displayed:
        1 -> Global acc
        2 -> Acc for each image
        3 -> TP, FP, TN, FN

    :param size:
    :param predicts: Predicted value for each spot (All images in the dataset)
    :param y_true: Real value for each spot (All images in the dataset)
    :return metrics: List containing all metrics results [Global Acc, [Acc per image], [TP, FP, TN, FN]]
    """

    metrics = []
    images_acc = []
    global_correct_label = 0
    image_correct_label = 0

    # True positive, False positive, True negative, False negative
    tp, fp, tn, fn = 0, 0, 0, 0

    # Reset generator
    test_data.load_custom(test_data.dataset_dir)
    last_image = 0
    total_spots = 0
    for _, xml_file in test_data.images:

        if type(xml_file) == list:
            pt_A, _, _, _, y = get_points_from_json(xml_file[1], xml_file[0], data_dir)
        else:
            pt_A, _, _, _, y = get_points_from_xml(xml_file)

        spot_number = len(pt_A)
        total_spots += spot_number
        predicts_in_use = predicts[last_image: last_image + spot_number]
        last_image += spot_number

        for j in range(len(predicts_in_use)):
            # Compare predicts to results and increase all related counters
            if predicts_in_use[j] == y[j]:
                global_correct_label += 1
                image_correct_label += 1

                # Evaluate True positive and True negative
                if predicts[j] == 1:
                    tp += 1
                else:
                    tn += 1
            else:

                # Evaluate False positive and False negative
                if y[j] == 1:
                    fn += 1
                else:
                    fp += 1

        images_acc.append(image_correct_label / spot_number)
        image_correct_label = 0

    metrics.append([global_correct_label / total_spots, 'Global Acc'])
    metrics.append([images_acc, 'Single image Acc'])
    metrics.append([[tp, fp, tn, fn], 'Confusion Matrix [TP, FP, TN, FN]'])
    metrics.append([tp / (tp + fp), "Precision"])
    metrics.append([tp / (tp + fn), "Recall"])
    metrics.append([(2*tp) / (2*tp + fp + fn),"F1-Score"])


    return metrics


def get_statistics(statistics, newPredicts, fps):
    """


    [[" [........], Ocupação atual"]
     [[.......], "Média de ocupação" ]
     [[......], [[..,..],[..., ...]], "Tempo médio de OCC"]
     [[dia, hora], "Dia e hora de pico", ]
    ]

    :param statistics:
    :param newPredicts:
    :param fps:
    :return:
    """

    # Ocupação atual
    new_occupation = sum(newPredicts) / len(newPredicts)
    statistics[0][0].append(new_occupation)

    # Média móvel
    if (len(statistics[1][0]) > 0):
        last_avg = statistics[1][0][len(statistics[1][0]) - 1]
        new_avg = (last_avg * len(statistics[1][0]) + new_occupation) / \
                  (len(statistics[1][0]) + 1)
    else:
        new_avg = new_occupation

    statistics[1][0].append(new_avg)

    # Tempo médio de ocupação
