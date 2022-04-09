import pandas as pd
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

def one_hot(lab, vals):
    hot = []
    for i in range(len(vals)):
        hot.append(lab.tolist().index(vals[i]))
    return hot

def data_gen():
    all_ann = pd.read_csv("datasets/LISA/allAnnotations.csv", delimiter=';')
    all_ann
    height = 224
    width = 224
    data = []
    labels = []
    path = "datasets/LISA/"
    test = 0
    import random
    sub_sample = random.sample(list(range(len(all_ann["Filename"]))),
                               k=len(all_ann["Filename"]) - 2000)  # The data set is too big for my PC
    for i in sub_sample:
        file = all_ann["Filename"][i]
        sign = all_ann["Annotation tag"][i]
        image = Image.open(path + file)
        image = image.resize((width, height))
        image = np.asarray(image)
        data.append(image)
        labels.append(sign)

    # 1k random samples of images with no signs
    with open("datasets/LISA/negatives/negatives.dat") as f:
        negatives = [line.rstrip('\n') for line in f]
    path_n = "datasets/LISA/negatives/"
    import random
    sub_sample = random.sample(list(range(len(negatives))), k=1000)
    for i in sub_sample:
        file = negatives[i]
        sign = "None"
        image = Image.open(path_n + file)
        image = image.resize((width, height))
        image = np.asarray(image)
        data.append(image)
        labels.append(sign)

    signs = np.array(data)
    labels = np.array(labels)

    # Randomize order
    s = np.arange(signs.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    signs = signs[s]
    labels = labels[s]

    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler
    X_train, X_test, y_train, y_test = train_test_split(signs, labels, test_size=0.2, random_state=1)
    del signs
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    # Preprocess
    X_train = X_train.astype('float64')
    X_val = X_val.astype('float64')
    X_test = X_test.astype('float64')

    # Normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Encode
    lab = np.unique(labels)
    y_train_e = np.array(one_hot(lab, y_train))
    y_val_e = np.array(one_hot(lab, y_val))
    y_test_e = np.array(one_hot(lab, y_test))

    tmp = pd.DataFrame()
    tmp["Label"] = lab
    tmp["Encoding"] = list(range(len(lab)))

    tmp.to_csv("datasets/key.csv", index=False)


    np.save("datasets/X_train1.npy", X_train)
    np.save("datasets/X_test1.npy", X_test)
    np.save("datasets/X_val1.npy", X_val)
    np.save("datasets/y_train_e1.npy", y_train_e)
    np.save("datasets/y_test_e1.npy", y_test_e)
    np.save("datasets/y_val_e1.npy", y_val_e)

data_gen()


