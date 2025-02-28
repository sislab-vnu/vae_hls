from fxpmath import Fxp
import random, warnings, glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import cv2

warnings.simplefilter(action='ignore', category=FutureWarning)
def seed_all(value=42):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)
    os.environ['PYTHONHASHSEED'] = str(value)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'

IMAGE_SIZE = (64, 64)

def get_subset_fixed_point(pathname, name=""):
    seed_all()
    images = []

    for fn in tqdm(glob.glob(pathname), desc=name):
        # Đọc ảnh và xử lý
        image = cv2.imread(fn, flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        images.append(image)  # Thêm ảnh vào danh sách
        # image_fixed_point = Fxp(image, like=fxp_ref)
        # images.append(image_fixed_point)

    return np.array(images)  # Trả về toàn bộ danh sách ảnh
def get_dataset(dataset_path):
    x_train = get_subset_fixed_point(os.path.join(dataset_path, 'train', 'good', '*.png'), 'Train images')
    x_good = get_subset_fixed_point(os.path.join(dataset_path, 'test', 'good', '*.png'), 'Good ')
    x_crack = get_subset_fixed_point(os.path.join(dataset_path, 'test', 'crack', '*.png'), 'crack')
    x_cut = get_subset_fixed_point(os.path.join(dataset_path, 'test', 'cut', '*.png'), 'cut')
    x_hole = get_subset_fixed_point(os.path.join(dataset_path, 'test', 'hole', '*.png'), 'hole')
    x_print = get_subset_fixed_point(os.path.join(dataset_path, 'test', 'print', '*.png'), 'print')
    
    print(f"x_train shape: {x_train.shape}")
    print(f"x_good sample fixed-point values:\n{x_good[0]}")
    # Kiểm tra kích thước của x_good
    print(f"x_good shape: {x_good.shape}")

    x_all = np.vstack((x_train, x_good, x_crack, x_cut, x_hole, x_print))

    # Shuffle dữ liệu
    x_all = x_all[np.random.permutation(len(x_all))]
    return x_all
