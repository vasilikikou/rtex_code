import os
import random
random.seed(42)
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input


def read_data(filepath, abnormal_only=False):
    '''

    :param filepath: Path of the data file
    :return: A dictionary with gold tags and a dictionary with image filenames
    '''
    exams_tags = {}
    exams_images = {}
    exams_captions = {}
    data = pd.read_csv(filepath, sep="\t")
    for index, row in data.iterrows():
        # Do not use exams with empty captions
        if not pd.isna(row.captions):
            # Use cases with two images
            if len(row.images.split(";")) == 2:
                if abnormal_only == False:
                    exams_tags[row.reports] = row.tags
                    exams_images[row.reports] = row.images
                    exams_captions[row.reports] = row.captions
                else:
                    if row.tags != "normal":
                        exams_tags[row.reports] = row.tags
                        exams_images[row.reports] = row.images
                        exams_captions[row.reports] = row.captions

    return exams_captions, exams_tags, exams_images

def load_image(image_path):
    '''

    :param image_path: The path of the image we want to load
    :return: A numpy array with the image
    '''
    # Load image with size 224x224
    img = image.load_img(image_path, target_size=(224, 224))
    # Turn the loaded image into a numpy array
    x = image.img_to_array(img)
    # Pre-process image for DenseNet
    x = preprocess_input(x)
    # Close
    img.close()

    return x

def encode_data_binary(images_path, batch_images, batch_tags):
    '''

    :param images_path: Path to the folder that contains the images
    :param batch_images: A list with the image filenames in the batch
    :param batch_tags: A list with the gold tags in the batch
    :return: Lists with the encoded images and the encoded gold input
    '''
    x1_data, x2_data, y_data = [], [], []

    for index, element in enumerate(batch_images):
        # Get the filenames of the two images
        images = element.split(";")
        # Load each image
        x1 = load_image(os.path.join(images_path, images[0]))
        x2 = load_image(os.path.join(images_path, images[1]))

        # normal = 0, abnormal = 1
        if batch_tags[index] == "normal":
            y = 0
        else:
            y = 1

        x1_data.append(x1)
        x2_data.append(x2)
        y_data.append(y)

    return x1_data, x2_data, y_data

def encode_data(images_path, batch_images, batch_tags, tags_list):
    '''

    :param images_path: Path to the folder that contains the images
    :param batch_images: A list with the image filenames in the batch
    :param batch_tags: A list with the gold tags in the batch
    :param tags_list: A list with all the available tags for classification
    :return: Lists with the encoded images and the encoded gold input
    '''
    x1_data, x2_data, y_data = [], [], []

    for index, element in enumerate(batch_images):
        # Get the filenames of the two images
        images = element.split(";")
        # Load each image
        x1 = load_image(os.path.join(images_path, images[0]))
        x2 = load_image(os.path.join(images_path, images[1]))
        # Create the ground truth vector with all zeros initially
        y = np.zeros(len(tags_list), dtype=float)
        # Get the gold tags
        tags = batch_tags[index].split(";")
        # For each possible tag if it appears in the ground truth...
        for i in range(0, len(tags_list)):
            # ...assign 1 to its position in the ground truth vector
            if tags_list[i] in tags:
                y[i] = 1

        x1_data.append(x1)
        x2_data.append(x2)
        y_data.append(y)

    return x1_data, x2_data, y_data