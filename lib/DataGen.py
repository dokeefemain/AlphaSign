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
    print(len(all_ann["Filename"]))
    all_ann = all_ann.drop_duplicates(subset=["Filename"], keep=False, ignore_index=True)
    print(len(all_ann["Filename"]))
    all_ann = all_ann[all_ann["Occluded,On another road"] != "1,0"]
    all_ann = all_ann.reset_index()

    height = 299
    width = 299

    data = []

    labels = []
    path = "datasets/LISA/"

    test = 0
    import random
    sub_sample = np.load("datasets/sample.npy")  # The data set is too big for my PC


    for i in range(len(all_ann["Filename"])):
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
    sub_sample = np.load("datasets/sample_negative.npy")
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
    img = Image.fromarray(X_test[0].astype('uint8'), 'RGB')
    img.save("img.png")

    # Normalize
    mean_image = np.mean(X_train, axis=0)
    np.save("datasets/mean.npy",mean_image)
    print(mean_image)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    img = Image.fromarray(X_test[0].astype('uint8'), 'RGB')
    img.save("img_norm.png")

    # Encode
    lab = np.unique(labels)
    y_train_e = np.array(one_hot(lab, y_train))
    y_val_e = np.array(one_hot(lab, y_val))
    y_test_e = np.array(one_hot(lab, y_test))

    tmp = pd.DataFrame()
    tmp["Label"] = lab
    tmp["Encoding"] = list(range(len(lab)))

    tmp.to_csv("datasets/keys_irv22.csv", index=False)


    np.save("datasets/X_train_irv22.npy", X_train)
    np.save("datasets/X_test_irv22.npy", X_test)
    np.save("datasets/X_val_irv22.npy", X_val)
    np.save("datasets/y_train_e_irv22.npy", y_train_e)
    np.save("datasets/y_test_e_irv22.npy", y_test_e)
    np.save("datasets/y_val_e_irv22.npy", y_val_e)

def data_gen_crop():
    all = pd.read_csv("datasets/LISA/allAnnotations.csv", delimiter=';')
    all_ann = all.drop_duplicates(subset=["Filename"], keep=False, ignore_index=True)

    all_ann = all_ann[all_ann["Occluded,On another road"] != "1,0"]
    all_ann = all_ann.reset_index()

    height = 299
    width = 299

    data = []

    labels = []
    path = "datasets/LISA/"

    test = 0
    import random
    sub_sample = np.load("datasets/sample.npy")  # The data set is too big for my PC

    for i in range(len(all_ann["Filename"])):
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
    sub_sample = np.load("datasets/sample_negative.npy")
    for i in sub_sample:
        file = negatives[i]
        sign = "None"
        image = Image.open(path_n + file)
        image = image.resize((width, height))
        image = np.asarray(image)
        data.append(image)
        labels.append(sign)

    df = all[all["Occluded,On another road"] == "0,0"]
    sample = random.sample(list(range(len(df["Filename"]))), k=1000)
    path = "datasets/LISA/"
    for i in sample:
        file = df["Filename"][i]
        sign = df["Annotation tag"][i]
        image = Image.open(path + file)
        rng = random.randint(2, 12)
        negative = Image.open("datasets/vid_negatives/Screenshot" + rng + ".png")
        n_size, i_size = negative.size, image.size
        bb = (df["Upper left corner X"][i], df["Upper left corner Y"][i], df["Lower right corner X"][i],
              df["Lower right corner Y"][i])
        tmp = image.crop(bb)
        x_r = n_size[0] / i_size[0]
        y_r = n_size[1] / i_size[1]
        reshape = (int(bb[2] * x_r - bb[0] * x_r), int(bb[2] * x_r - bb[0] * x_r))
        tmp = tmp.resize(reshape)
        negative.paste(tmp, (int(bb[0] * x_r), int(bb[1] * y_r)))
        negative.resize((width, height))
        image = np.asarray(negative)
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
    img = Image.fromarray(X_test[0].astype('uint8'), 'RGB')
    img.save("img.png")

    # Normalize
    mean_image = np.mean(X_train, axis=0)
    np.save("datasets/mean.npy", mean_image)
    print(mean_image)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    img = Image.fromarray(X_test[0].astype('uint8'), 'RGB')
    img.save("img_norm.png")

    # Encode
    lab = np.unique(labels)
    y_train_e = np.array(one_hot(lab, y_train))
    y_val_e = np.array(one_hot(lab, y_val))
    y_test_e = np.array(one_hot(lab, y_test))

    tmp = pd.DataFrame()
    tmp["Label"] = lab
    tmp["Encoding"] = list(range(len(lab)))

    tmp.to_csv("datasets/keys_crop.csv", index=False)

    np.save("datasets/X_train_crop.npy", X_train)
    np.save("datasets/X_test_crop.npy", X_test)
    np.save("datasets/X_val_crop.npy", X_val)
    np.save("datasets/y_train_e_crop.npy", y_train_e)
    np.save("datasets/y_test_e_crop.npy", y_test_e)
    np.save("datasets/y_val_e_crop.npy", y_val_e)

from sklearn.preprocessing import LabelBinarizer
def data_gen_bb():
    Adata = pd.read_csv("datasets/LISA/allAnnotations.csv", delimiter=';')
    height = 227
    width = 227
    data = []
    labels = []
    bboxes = []
    imagePaths = []
    path = "datasets/LISA/"
    test = 0
    import random
    sub_sample = random.sample(list(range(len(Adata["Filename"]))), k=len(Adata["Filename"]) - 1000)
    for i in range(len(Adata["Filename"])):
        file = Adata["Filename"][i]
        sign = Adata["Annotation tag"][i]
        image = Image.open(path + file)
        h, w = image.height, image.width
        image = image.resize((width, height))
        image = np.asarray(image)
        data.append(image)
        labels.append(sign)
        bboxes.append((float(Adata['Upper left corner X'][i])/w,float(Adata['Upper left corner Y'][i])/h,float(Adata['Lower right corner X'][i])/w,float(Adata['Lower right corner Y'][i])/h))
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    from sklearn.model_selection import train_test_split
    split = train_test_split(data, labels, bboxes, test_size=0.20)
    (X_train, X_test) = split[:2]
    (y_train, y_test) = split[2:4]
    (z_train, z_test) = split[4:6]
    split = train_test_split(X_train, y_train, z_train, test_size=0.25)
    (X_train, X_val) = split[:2]
    (y_train, y_val) = split[2:4]
    (z_train, z_val) = split[4:6]




    np.save("datasets/X_train_bb.npy", X_train)
    np.save("datasets/X_test_bb.npy", X_test)
    np.save("datasets/X_val_bb.npy", X_val)
    np.save("datasets/y_train_e_bb.npy", y_train)
    np.save("datasets/y_test_e_bb.npy", y_test)
    np.save("datasets/y_val_e_bb.npy", y_val)
    np.save("datasets/z_train_bb.npy", z_train)
    np.save("datasets/z_test_bb.npy", z_test)
    np.save("datasets/z_val_bb.npy", z_val)

data_gen_crop()


