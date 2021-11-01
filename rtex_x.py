import os
import numpy as np
import pandas as pd
import data.data_loading as dl

from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from evaluation import caption_evaluate as c_eval
from tensorflow.keras.applications.densenet import preprocess_input

def extract_img_embeddings(model, images_path, data):
    '''

    :param model: CNN model to extract embeddings from
    :param images_path: The path to the folder that contains the images
    :param data: Dictionary with exam ids as keys and and image filenames as values
    :return: A dictionary with the exam ids as keays and extracted image embeddings as values
    '''
    exams_vectors = {}
    for report in tqdm(data):
        # Get the two image filenames
        images = data[report].split(";")
        encoded = []
        for i in images:
            # Load and encode image
            image_path = os.path.join(images_path, i)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            encoded.append(x)
            img.close()

        # Feed the images to the model and get the exam embedding
        x1 = np.expand_dims(encoded[0], axis=0)
        x2 = np.expand_dims(encoded[1], axis=0)
        vector = model.predict([x1,x2])

        exams_vectors[report] = vector.transpose().flatten()
    return exams_vectors

def nn(vec, train_image_mat, train_exams_ids):
    assert train_image_mat.shape[0] == len(train_exams_ids)
    vec = vec / np.sum(vec)
    vec_clones = np.array([vec] * train_image_mat.shape[0])
    similarities = np.sum(vec_clones * train_image_mat, 1)
    nearest_id = train_exams_ids[np.argmax(similarities)]
    return nearest_id

def run_rtex_x(train_path, test_path, images_path, results_path):
    # Load the pre-trained RTEx@T model
    classification_model = load_model(os.path.join(results_path, "rtex_t.hdf5"))
    # Choose the layer from which to extract image embeddings
    vector_extraction_model = Model(inputs=classification_model.input,
                                    outputs=classification_model.get_layer("concatenate_embeddings").output)

    print("Loaded model to extract to exam embeddings")

    # Read training data
    train_exams_captions, train_exams_tags, train_exams_images = dl.read_data(train_path, abnormal_only=True)

    print("Extracting training exam embeddings...")

    # Get the exam embeddings for training
    train_images_vec = extract_img_embeddings(vector_extraction_model, images_path, train_exams_images)

    # Read test data
    test_exams_captions, test_exams_tags, test_exams_images = dl.read_data(test_path, abnormal_only=True)

    # Get the ids of the test exams
    test_exams = list(test_exams_captions.keys())

    print("Extracting test exam embeddings...")

    # Get the exam embeddings for test
    test_images_vec = extract_img_embeddings(vector_extraction_model, images_path, test_exams_images)

    # Read tsv file with predicted tags for each exam
    res = pd.read_csv(os.path.join(results_path, "rtex_t_predictions.tsv"), sep="\t",
                      names=["exams","tags"], header=None)

    # Replace NaNs with none tag
    res["tags"] = res["tags"].replace(np.nan, "none")
    test_cases_predicted_tags = dict(zip(res.exams, res.tags))

    print("Calculating most similar training exam for each test exam...")

    # 1NN+
    sim_test_results = {}
    for test_id in tqdm(test_exams_captions.keys()):
        # Filter the training database for exams with the same predicted tags (if none, use all)
        predicted_tags = ";".join(sorted(test_cases_predicted_tags[test_id].split(";")))
        train_exams_db = [exam for exam in train_exams_tags.keys()
                          if predicted_tags == ";".join(sorted(train_exams_tags[exam].split(";")))]
        if len(train_exams_db) == 0 :
            train_exams_db = list(train_images_vec.keys())
        # Compute dot similarity with the exams in the training database (filtered or not)
        raw = np.array([train_images_vec[i] for i in train_exams_db])
        raw = raw / np.array([np.sum(raw,1)] * raw.shape[1]).transpose()
        sim_test_results[test_id] = train_exams_captions[nn(test_images_vec[test_id], raw, train_exams_db)]

    # Save test predictions to tsv file
    df = pd.DataFrame.from_dict(sim_test_results, orient="index")
    df.to_csv(os.path.join(results_path, "rtex_x.tsv"), sep="\t", header=False)

    # Save the predicted captions in CheXpert format
    captions = list(sim_test_results.values())
    unique_captions_df = pd.DataFrame(set(captions))
    unique_captions_df.to_csv(os.path.join(results_path, "rtex_x.csv"), header=False, index=False)

    print("Results saved")

    print("Evaluation...")
    gts, res = c_eval.prepare_captions(test_exams_captions, sim_test_results)
    c_eval.compute_scores(gts, res)