import os
import random
random.seed(42)
import numpy as np
import pandas as pd
import data.data_loading as dl

from tqdm import tqdm
from math import ceil
from numpy.random import seed
seed(42)
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from evaluation.tagging_evaluation import evaluate_f1
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau


def train_generator(train_tags, train_images, tags_list, batch_size, images_path):
    '''

    :param train_tags: Dictionary with the exam ids as keys and the gold tags as values
    :param train_images: Dictionary with the exam ids as keys and the image filenames as values
    :param tags_list: A list with all the available tags for classification
    :param batch_size: Size of each batch
    :return: An encoded input batch
    '''
    while True:
        exam_ids = list(train_tags.keys())
        random.shuffle(exam_ids)
        # Loop over all exams with batch size as the step
        for i in range(0, len(exam_ids), batch_size):
            x_images, x_tags = list(), list()
            # Loop over the exams in the batch and yield their instances
            for j in range(i, min(len(exam_ids), i + batch_size)):
                # Get the id of the exam
                exam_id = exam_ids[j]
                # Get exam's images
                images = train_images[exam_id]
                # Get exam's gold tags
                tags = train_tags[exam_id]
                # Add them to the batch
                x_images.append(images)
                x_tags.append(tags)

            encoded_x1, encoded_x2, encoded_y = dl.encode_data(images_path, x_images, x_tags, tags_list)

            yield ([np.array(encoded_x1), np.array(encoded_x2)], np.array(encoded_y))

def val_generator(val_tags, val_images, tags_list, batch_size, images_path):
    while True:
        # Loop over all images
        case_ids = list(val_tags.keys())
        random.shuffle(case_ids)
        for i in range(0, len(case_ids), batch_size):
            x_images, x_tags = list(), list()
            # Loop over the images in the batch and yield their instances
            for j in range(i, min(len(case_ids), i + batch_size)):
                case_id = case_ids[j]
                # Get case's images
                images = val_images[case_id]
                # Get image caption
                tags = val_tags[case_id]

                x_images.append(images)
                x_tags.append(tags)

            encoded_x1, encoded_x2, encoded_y = dl.encode_data(images_path, x_images, x_tags, tags_list)

            yield ([np.array(encoded_x1), np.array(encoded_x2)], np.array(encoded_y))

def create_model(encoder, num_tags):
    '''

    :param encoder: Pre-trained CNN model for encoding the images
    :param num_tags: Number of possible tags for classification
    :return: The built model
    '''

    # Define the layer from which to extract the encoder output
    x = encoder.get_layer("avg_pool").output
    y = encoder.get_layer("relu").output
    # Build the encoder sub-model
    cnn_model = Model(inputs=encoder.input, outputs=[x, y])

    # Image inputs
    img_input1 = Input(shape=(224, 224, 3), name="img_input1")
    img_input2 = Input(shape=(224, 224, 3), name="img_input2")
    # Feed the image to the encoder sub-model
    img_emb1 = cnn_model(img_input1)[0]
    img_emb2 = cnn_model(img_input2)[0]
    # Concatenate the two image embeddings
    mean_emb = Concatenate(name="concatenate_embeddings")([img_emb1, img_emb2])
    # Classifier layer
    outputs = Dense(num_tags, activation="sigmoid", name="classifier", kernel_initializer=glorot_uniform(seed=42))(
        mean_emb)
    # This is the model we will train
    model = Model(inputs=[img_input1,img_input2], outputs=outputs)

    return model

def run_rtex_t(train_path, val_path, test_path, images_path, tags_path, results_path):
    # Read all tags available in the dataset
    tags_df = pd.read_csv(tags_path, header=None)
    tags_list = tags_df[0].to_list()
    num_tags = len(tags_list)

    # Read data
    _, train_cases_tags, train_cases_images = dl.read_data(train_path, abnormal_only=True)
    _, val_cases_tags, val_cases_images = dl.read_data(val_path, abnormal_only=True)

    print("Loaded data. Starting training...")

    # Define pre-trained CNN encoder
    base_model = DenseNet121(weights='imagenet', include_top=True)

    # Create RTEx@T model
    model = create_model(base_model, num_tags)

    # Compila
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

    # Add early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="auto", restore_best_weights=True)

    csv_logger = CSVLogger(os.path.join(results_path, "rtex_t.log"))

    # Save best model checkpoint
    checkpoint = ModelCheckpoint(os.path.join(results_path, "rtex_t.hdf5"), monitor="val_loss",
                                 save_best_only=True, mode="auto")

    # Reduce learning rate mechanism
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode="min")

    batch_size = 16
    epochs = 100

    print("Starting training...")

    ### Training ###
    history = model.fit(
        train_generator(train_cases_tags, train_cases_images, tags_list, batch_size, images_path),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stopping, checkpoint, csv_logger, reduce_lr],
        validation_data=val_generator(val_cases_tags, val_cases_images, tags_list, batch_size, images_path),
        steps_per_epoch=ceil(len(train_cases_images) / batch_size),
        validation_steps=ceil(len(val_cases_images) / batch_size))

    ### Inference ###

    print("Inference step. Predicting labels...")

    _, test_cases_tags, test_cases_images = dl.read_data(test_path, abnormal_only=True)

    # Encode val images
    x_val1, x_val2, _ = dl.encode_data(images_path, list(val_cases_images.values()), list(val_cases_tags.values()),
                                       tags_list)
    print("Encoded validation images")

    # Encode test images
    x_test1, x_test2, _ = dl.encode_data(images_path, list(test_cases_images.values()), list(test_cases_tags.values()),
                                         tags_list)
    print("Encoded test images")

    predictions = model.predict([np.array(x_val1), np.array(x_val2)], batch_size=16, verbose=1)
    print("Got predictions for val set")

    # Tuning
    f1_scores = {}
    steps = 1000

    print("Evaluating different thresholds")
    for i in tqdm(range(steps)):
        threshold = float(i) / steps
        y_pred = {}
        for j in range(len(predictions)):
            predicted_tags = []
            indices = np.argwhere(predictions[j] >= threshold).flatten()
            for index in indices:
                predicted_tags.append(tags_list[index])
            y_pred[list(val_cases_tags.keys())[j]] = ";".join(predicted_tags)

        f1_scores[threshold], _, _ = evaluate_f1(val_cases_tags, y_pred)

    # Find the best f1
    best_threshold = max(f1_scores, key=f1_scores.get)

    print(
        "The best F1 score is " + str(f1_scores[best_threshold]) + " achieved with threshold = " + str(best_threshold))

    # Get predictions for test set
    test_predictions = model.predict([np.array(x_test1), np.array(x_test2)], batch_size=16, verbose=1)

    results = {}
    for i in range(len(test_predictions)):
        predicted_tags = []
        for j in range(len(tags_list)):
            if test_predictions[i, j] >= best_threshold:
                predicted_tags.append(tags_list[j])
        results[list(test_cases_tags.keys())[i]] = ";".join(predicted_tags)

    with open(os.path.join(results_path, "rtex_t_predictions.tsv"), "w") as output_file:
        for image in results:
            output_file.write(str(image) + "\t" + results[image])
            output_file.write("\n")

    f1, p, r = evaluate_f1(test_cases_tags, results)
    print("*" * 100)
    print("Test results:")
    print("F1 =", f1)
    print("Precision =", p)
    print("Recall =", r)