import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from model import build_classifier
from metrics import sensitivity, specificity


H = 224
W = 224
RNG_SEED = 42

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)

    project_path = "drive/MyDrive/DIP Project"
    classification_path = f"{project_path}/Classification"
    files_path = f"{classification_path}/Classification Files"

    """ Directory for storing files """
    create_dir(files_path)

    """ Hyperparameters """
    batch_size = 8
    lr = 1e-4
    num_epochs = 5
    model_path = f"{files_path}/model.h5"
    csv_path = f"{files_path}/data.csv"

    """ Dataset """
    data_path = f"{project_path}/For Project/Classification"
    npy_path = f"{data_path}/NPYs"
    
    training_images = np.load(f"{npy_path}/Images.npy")
    training_labels = np.load(f"{npy_path}/Labels.npy")


    training_path = f"{data_path}/Training"
    # testing_path = f"{data_path}/Validation"
    
    class_path = glob(f"{training_path}/*")

    """ Model """
    model = build_classifier((H, W, 3), len(class_path))
    metrics = ['categorical_accuracy', sensitivity, specificity]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        x = training_images,
        y = training_labels,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=False
    )