import numpy as np
import cv2
import os
from glob import glob
from sklearn.utils import shuffle

H = 224
W = 224
RNG_SEED = 42

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=RNG_SEED)
    return x, y

if __name__ == "__main__":
    project_path = "drive/MyDrive/DIP Project"
    data_path = f"{project_path}/For Project/Classification"
    npy_path = f"{data_path}/NPYs"

    create_dir(npy_path)


    testing_path = f"{data_path}/Validation"
    class_path = glob(f"{testing_path}/*")

    x_npy = []
    y_npy = []
    for i in range(len(class_path)):
        img_list = glob(f"{class_path[i]}/*.jpg")
        y = np.zeros((len(class_path)), dtype=np.float32)
        y[i] = 1.0
        for img_name in img_list:
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (H, W))
            img = img / 255.0
            img = img.astype(np.float32)
            x_npy.append(img)
            y_npy.append(y)

    x_npy = np.array(x_npy)
    y_npy = np.array(y_npy)
    print(x_npy.shape)
    print(y_npy.shape)

    x_npy, y_npy = shuffling(x_npy, y_npy)

    np.save(f"{npy_path}/Images_testing.npy", x_npy)
    np.save(f"{npy_path}/Labels_testing.npy", y_npy)
        
