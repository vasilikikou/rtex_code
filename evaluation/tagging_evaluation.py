import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_f1(gt_pairs, pr_pairs):
    '''

    :param gt_pairs: Dictionary with gold tags
    :param pr_pairs: Dictionary with produced tags
    :return: F1, precision and recall scores
    '''

    f1_scores = 0
    precisions = 0
    recalls = 0

    # Check the exam ids of candidate and ground truth are the same
    assert list(pr_pairs.keys()) == list(gt_pairs.keys())

    # Evaluate the predicted tags against the gold tags for each exam
    for exam_id in pr_pairs:
        if pd.isna(pr_pairs[exam_id]):
            candidate_tags = ""
        else:
            candidate_tags = pr_pairs[exam_id].upper()

        if pd.isna(gt_pairs[exam_id]):
            gt_tags = ""
        else:
            gt_tags = gt_pairs[exam_id].upper()

        # Split concept string into concept array
        # Manage empty concept lists
        if gt_tags.strip() == "" or gt_tags.strip() == "NORMAL":
            gt_tags = []
        else:
            gt_tags = gt_tags.split(';')

        if candidate_tags.strip() == "" or candidate_tags.strip() == "NORMAL":
            candidate_tags = []
        else:
            candidate_tags = candidate_tags.split(';')

        if len(gt_tags) == 0 and len(candidate_tags) == 0:
            f1 = 1
            p = 1
            r = 1
        else:
            # Global set of concepts
            all_tags = sorted(list(set(gt_tags + candidate_tags)))

            # Calculate F1 score for the current concepts
            y_true = [int(tag in gt_tags) for tag in all_tags]
            y_pred = [int(tag in candidate_tags) for tag in all_tags]

            f1 = f1_score(y_true, y_pred, average='binary')
            p = precision_score(y_true, y_pred, average="binary")
            r = recall_score(y_true, y_pred, average="binary")

        # Increase calculated score
        f1_scores += f1
        precisions += p
        recalls += r

    mean_f1_score = f1_scores / len(gt_pairs)
    mean_precision_score = precisions / len(gt_pairs)
    mean_recall_score = recalls / len(gt_pairs)

    return mean_f1_score, mean_precision_score, mean_recall_score
