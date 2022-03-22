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

    np.save("datasets/X_train.npy", X_train)
    np.save("datasets/X_test.npy", X_test)
    np.save("datasets/X_val.npy", X_val)
    np.save("datasets/y_train_e.npy", y_train_e)
    np.save("datasets/y_test_e.npy", y_test_e)
    np.save("datasets/y_val_e.npy", y_val_e)



def data_gen_bb():

    all_ann = pd.read_csv("datasets/LISA/allAnnotations.csv", delimiter=';')
    all_ann
    height = 150
    width = 150
    data = []

    bboxes = []
    labels = []

    path = "datasets/LISA/"
    test = 0
    import random
    sub_sample = random.sample(list(range(len(all_ann["Filename"]))),
                               k=len(all_ann["Filename"]) - 1000)  # The data set is too big for my PC
    np.save("datasets/sample.npy",sub_sample)
    for i in sub_sample:
        file = all_ann["Filename"][i]
        sign = all_ann["Annotation tag"][i]

        image = Image.open(path + file)
        h, w = image.height, image.width
        image = image.resize((width, height))
        image = np.asarray(image)

        data.append(image)
        labels.append(sign)
        bboxes.append((float(all_ann['Upper left corner X'][i]) / w, float(all_ann['Upper left corner Y'][i]) / h,
                       float(all_ann['Lower right corner X'][i]) / w, float(all_ann['Lower right corner Y'][i]) / h))


    signs = np.array(data)
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")

    # Randomize order
    s = np.arange(signs.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    signs = signs[s]
    labels = labels[s]
    mean_image = np.mean(signs, axis=0)

    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler
    split = train_test_split(signs, labels, bboxes, test_size=0.25, random_state=42)
    del signs
    # (train_signs, test_signs) = split[:2]
    # (train_labels, test_labels) = split[2:4]
    # (train_bboxes, test_bboxes) = split[4:]
    # split = train_test_split(train_signs, train_labels, train_bboxes, test_size=0.25, random_state=42)
    (train_signs, val_signs) = split[:2]
    (train_labels, val_labels) = split[2:4]
    (train_bboxes, val_bboxes) = split[4:]


    # Preprocess
    train_signs = train_signs.astype('float64')
    val_signs = val_signs.astype('float64')
    #test_signs = test_signs.astype('float64')

    # Normalize


    train_signs /= 150
    val_signs /= 150
    #test_signs /= 225

    # Encode
    lab = np.unique(labels)
    train_labels_e = np.array(one_hot(lab, train_labels))
    val_labels_e = np.array(one_hot(lab, val_labels))
    #test_labels_e = np.array(one_hot(lab, test_labels))

    tmp = pd.DataFrame()
    tmp["Label"] = lab
    tmp["Encoding"] = list(range(len(lab)))

    tmp.to_csv("datasets/bbkey.csv", index=False)

    np.save("datasets/train_signs3k.npy", train_signs)
    #np.save("datasets/test_signs.npy", test_signs)
    np.save("datasets/val_signs3k.npy", val_signs)
    np.save("datasets/train_labels_e3k.npy", train_labels_e)
    #np.save("datasets/test_labels_e.npy", test_labels_e)
    np.save("datasets/val_labels_e3k.npy", val_labels_e)
    np.save("datasets/train_bboxes3k.npy", train_bboxes)
    #np.save("datasets/test_bboxes.npy", test_bboxes)
    np.save("datasets/val_bboxes3k.npy", val_bboxes)

data_gen_bb()


