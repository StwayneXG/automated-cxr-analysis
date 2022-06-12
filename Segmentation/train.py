import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from model import build_unet
from metrics import dice_loss, dice_coef

H = 512
W = 512
RNG_SEED = 42

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=RNG_SEED)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "images", "*.png")))
    y = sorted(glob(os.path.join(path, "masks", "*.png")))
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (H, W))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (H, W))
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(RNG_SEED)
    tf.random.set_seed(RNG_SEED)

    project_path = "drive/MyDrive/DIP Project"
    data_path = f"{project_path}/For Project/Segmentation"
    segmentation_path = f"{project_path}/Segmentation"

    """ Directory for storing files """
    create_dir(f"{segmentation_path}/Segmentation Files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 5
    model_path = f"{segmentation_path}/Segmentation Files/model.h5"
    csv_path = f"{segmentation_path}/Segmentation Files/data.csv"

    """ Dataset """
    training_path = f"{data_path}/Training Data"

    data_x, data_y = load_data(training_path)
    data_x, data_y = shuffling(data_x, data_y)
    
    train_size = int(0.8 * len(data_x))
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]

    valid_x = data_x[train_size:]
    valid_y = data_y[train_size:]

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet((H, W, 1))
    metrics = [dice_coef]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
    )