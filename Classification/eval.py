import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from metrics import sensitivity, specificity

H = 224
W = 224

RNG_SEED = 42

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)

    project_path = "drive/MyDrive/DIP Project"
    data_path = f"{project_path}/For Project/Classification"
    classification_path = f"{project_path}/Classification"

    """ Directory for storing files """
    create_dir(f"{classification_path}/Results")

    """ Loading model """
    with CustomObjectScope({'sensitivity': sensitivity, 'specificity': specificity}):
        model = tf.keras.models.load_model(f"{classification_path}/Archive/Classification Files [Stacked Model (MobileNetV2 + DenseNet 169)]/model.h5")
        # model = tf.keras.models.load_model(f"{classification_path}/Classification Files/model.h5")

    """ Load the dataset """
    npy_path = f"{data_path}/NPYs"
    testing_file_path = f"{npy_path}/Images_testing.npy"
    testing_labels_path = f"{npy_path}/Labels_testing.npy"
    
    testing_images, testing_labels = np.load(testing_file_path), np.load(testing_labels_path) 

    """ Evaluation and Prediction """
    
    my_predictions = model.predict(testing_images)
    print(my_predictions.shape)


    cmd = ConfusionMatrixDisplay((confusion_matrix(list(np.argmax(testing_labels, axis=1)), list(np.argmax(my_predictions, axis=1)))), display_labels=['ATELECTASIS', 'PLEURAL EFFUSION', 'CARDIOMEGALY', 'CONSOLIDATION', 'EDEMA', 'NO FINDING'])
    cmd.plot()
    plt.show()