import argparse

from rtex_r import run_rtex_r
from rtex_t import run_rtex_t
from rtex_x import run_rtex_x


parser = argparse.ArgumentParser(description="Train and predict for each step of RTEx")
parser.add_argument("--train_path", help="The tsv file of the training set")
parser.add_argument("--val_path", help="The tsv file of the validation set")
parser.add_argument("--test_path", help="The tsv file of the test set")
parser.add_argument("--images_path", help="The path to the folder that contains the images")
parser.add_argument("--results_path", help="The path to the folder to save the results files")
parser.add_argument("--tags_path", help="The csv file with all the available tags for classification")


if __name__ == "__main__":
    args = parser.parse_args()
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    images_path = args.images_path
    results_path = args.results_path
    tags_path = args.tags_path

    # Step 1 - Ranking
    print("Ranking...")
    run_rtex_r(train_path, val_path, test_path, images_path, results_path)
    print("*" * 100)
    # Step 2 - Tagging
    print("Tagging...")
    run_rtex_t(train_path, val_path, test_path, images_path, tags_path, results_path)
    print("*" * 100)
    # Step 3 - Captioning
    print("Captioning...")
    run_rtex_x(train_path, test_path, images_path, results_path)