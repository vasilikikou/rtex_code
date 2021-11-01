import os
import random
random.seed(42)
import numpy as np
import pandas as pd
import data.data_loading as dl

from math import ceil
from tqdm import tqdm
from numpy.random import seed
seed(42)
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import f1_score, precision_score, recall_score


def train_generator(train_tags, train_images, batch_size, images_path):
    '''

    :param train_tags: Dictionary with the exam ids as keys and the gold tags as values
    :param train_images: Dictionary with the exam ids as keys and the image filenames as values
    :param batch_size: Size of each batch
    :param images_path: Path to the folder that contains the images
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
            # Encode inputs
            encoded_x1, encoded_x2, encoded_y = dl.encode_data_binary(images_path, x_images, x_tags)

            yield ([np.array(encoded_x1), np.array(encoded_x2)], np.array(encoded_y))

def val_generator(val_tags, val_images, batch_size, images_path):
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

            encoded_x1, encoded_x2, encoded_y = dl.encode_data_binary(images_path, x_images, x_tags)

            yield ([np.array(encoded_x1), np.array(encoded_x2)], np.array(encoded_y))

def run_rtex_r(train_path, val_path, test_path, images_path, results_path):
    # Read data from files
    _, train_cases_tags, train_cases_images = dl.read_data(train_path)
    _, val_cases_tags, val_cases_images = dl.read_data(val_path)

    print("Loaded data. Building model...")

    # Load pre-trained DenseNet
    base_model = DenseNet121(weights='imagenet', include_top=True)
    # Extract image embedding from last average pooling layer
    x = base_model.get_layer("avg_pool").output
    # Build CNN encoder component
    cnn_model = Model(inputs=base_model.input, outputs=x)

    # Image inputs
    img_input1 = Input(shape=(224,224,3), name="img_input1")
    img_input2 = Input(shape=(224,224,3), name="img_input2")
    # Get image embeddings
    img_emb1 = cnn_model(img_input1)
    img_emb2 = cnn_model(img_input2)
    # Concatenate the two embeddings
    concat_emb = Concatenate()([img_emb1, img_emb2])
    # Add the classifier layer
    abnormality_prob = Dense(1, activation="sigmoid", name="classifier", kernel_initializer=glorot_uniform(seed=42))\
        (concat_emb)

    # Build final model
    model = Model(inputs=[img_input1,img_input2], outputs=abnormality_prob)

    # Compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

    # Add early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="auto", restore_best_weights=True)

    csv_logger = CSVLogger(os.path.join(results_path, "rtex_r.log"))

    # Save best model checkpoint
    checkpoint = ModelCheckpoint(os.path.join(results_path, "rtex_r_checkpoint.hdf5"), monitor="val_loss",
                                 save_best_only=True, mode="auto")
    # Reduce learning rate mechanism
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode="min")

    # Define batch size and epochs
    batch_size = 16
    epochs = 100

    print("Starting training...")

    ### Training ###
    history = model.fit(
        train_generator(train_cases_tags, train_cases_images, batch_size, images_path),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, checkpoint, csv_logger, reduce_lr],
        validation_data=val_generator(val_cases_tags, val_cases_images, batch_size, images_path),
        steps_per_epoch=ceil(len(train_cases_images) / batch_size),
        validation_steps=ceil(len(val_cases_images) / batch_size))

    ### Inference ###

    print("Inference step. Predicting labels...")

    # Read test data from files
    _, test_cases_tags, test_cases_images = dl.read_data(test_path)
    # Encode
    x_test1, x_test2, y_true = dl.encode_data_binary(images_path, list(test_cases_images.values()),
                                                     list(test_cases_tags.values()))

    # Get predictions on test set
    test_predictions2 = model.predict([np.array(x_test1), np.array(x_test2)], batch_size=16, verbose=1)

    # Get predicted labels
    results = {}
    for i, prob in tqdm(enumerate(test_predictions2)):
        # If abnormality probability is above 0.5, label the exam as abnormal
        if prob > 0.5:
            results[list(test_cases_images.keys())[i]] = 1
        else:
            results[list(test_cases_images.keys())[i]] = 0

    print("Evaluation...")
    precision = precision_score(np.array(y_true), np.array(list(results.values())))
    print("Precision score:", round(precision, 3))

    recall = recall_score(np.array(y_true), np.array(list(results.values())))
    print("Recall score:", round(recall, 3))

    f1 = f1_score(np.array(y_true), np.array(list(results.values())))
    print("F1 score:", round(f1, 3))

    # Save results
    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv(os.path.join(results_path, "rtex_r_results.tsv"), sep="\t", header=False)