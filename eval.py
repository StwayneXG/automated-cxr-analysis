import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from metrics import dice_loss, dice_coef
from train import load_data

H = 512
W = 512

RNG_SEED = 42

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    ## img - mask - pred
    line = np.ones((H, 10, 3)) * 128

    """ Image """
    image = np.expand_dims(image, axis=-1)    ## (512, 512, 1)
    image = np.concatenate([image, image, image], axis=-1)  ## (512, 512, 3)

    """ Mask """
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)

    """ Predicted Mask """
    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)

    project_path = "drive/MyDrive/DIP Project"
    data_path = f"{project_path}/For Project/Segmentation"
    segmentation_path = f"{project_path}/Segmentation"

    """ Directory for storing files """
    create_dir(f"{segmentation_path}/Results")

    """ Loading model """
    with CustomObjectScope({'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(f"{segmentation_path}/Segmentation Files/model.h5")

    """ Load the dataset """
    test_x = sorted(glob(f"{data_path}/Validation Data/images/*"))
    test_y = sorted(glob(f"{data_path}/Validation Data/masks/*"))
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(H, W))
        x = image/255.0
        """ Adding Channel Dimension """
        x = np.expand_dims(x, axis=-1)  
        """ Adding Tensor Dimension """
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=(H, W))
        y = mask/255.0
        y = y > 0.5
        y = y.astype(np.int32)

        """ Prediction """
        y_pred = model.predict(x)[0]
        """ Removing Channel Dimension """
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = f"{segmentation_path}/Results/{name}.png"
        save_results(image, mask, y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        SCORE.append([name, acc_value, f1_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precision: {score[3]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Recall", "Precision"])
    df.to_csv("drive/MyDrive/DIP Project/Segmentation/Segmentation Files/score.csv")