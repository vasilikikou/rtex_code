# RTEX: A novel framework for ranking, tagging, and explanatory diagnostic captioning of radiography exams

This repository contains the code for our framework [RTEx](https://academic.oup.com/jamia/article/28/8/1651/6242739?login=true), 
which is published in the Journal of the American Medical Informatics Association.

## Prerequisites

Create the conda environment containing the packages required to run the code 
with the following command:
```shell
conda env create -f environment.yml
```
To activate the environment:
```shell
conda activate rtex
```

## Run RTEx

In order to run training and inference for all the steps of RTEx run the `run_rtex.py` script, providing the paths to datafiles 
(training, validation, test and tags), folder with images and folder to save the results. For more details run 
```shell
python run_rtex.py -h
``` 
The data should be provided in tab-separated files with columns: 
* `reports`: A unique id for each exam.
* `images`: The two images (filenames) that correspond to each exam separated by ";", *e.g. "image1.jpg;image2.jpg"*.
* `captions`: The diagnostic text for each exam.
* `tags`: The assigned tags for each exam. If there are more than one tags assigned, they should be separated by ";", *e.g. "tag1.jpg;tag2.jpg"*.

In case of exams with no abnormalities, the value in the `tags` column should be **"normal"**.