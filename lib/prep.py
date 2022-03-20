import pandas as pd
import numpy as np
from PIL import Image
from keras.layers.wrappers import TimeDistributed
def data_prep(path_csv, path, height, width, time_steps = 0, model = None, rnn = False, rnn1 = False, path_model = ""):
    vid = pd.read_csv(path_csv, delimiter=';')
    data = []
    labels = []
    test = 0
    for i in range(len(vid["Filename"])):
        file = vid["Filename"][i]
        sign = vid["Annotation tag"][i]
        image = Image.open(path + file)
        image = image.resize((width, height))
        image = np.asarray(image)
        data.append(image)
        labels.append(sign)
    signs = np.array(data)
    labels = np.array(labels)
    signs = signs.astype('float64')
    key = pd.read_csv("App/lib/datasets/key2.csv")
    key = key.groupby("Label")
    encodings = []
    for i in labels:
        ind = key.get_group(i)["Encoding"]
        tmp = np.zeros(48)
        tmp[ind] = 1
        encodings.append(list(tmp))



    signs -= np.mean(signs)

    if rnn:
        pred = model.predict(signs)
        total = len(pred)
        while total % time_steps != 0:
            total -= 1
        loops = total / time_steps
        X = []
        tmp = []
        y = []
        for i in range(total):
            tmp.append(pred[i])
            if (i + 1) % time_steps == 0:
                X.append(tmp)
                y.append(encodings[i])
                tmp = []
        return signs, labels, np.array(X), np.array(y)

def prep_rnn(path_csv, path, height, width,model, time_steps):
    vid = pd.read_csv(path_csv, delimiter=';')
    data = []
    labels = []
    test = 0
    for i in range(len(vid["Filename"])):
        file = vid["Filename"][i]
        sign = vid["Annotation tag"][i]
        image = Image.open(path + file)
        image = image.resize((width, height))
        image = np.asarray(image)
        data.append(image)
        labels.append(sign)
    signs = np.array(data)
    labels = np.array(labels)
    signs = signs.astype('float64')
    key = pd.read_csv("App/lib/datasets/key1.csv")
    key = key.groupby("Label")
    encodings = []
    for i in labels:
        ind = key.get_group(i)["Encoding"]
        tmp = np.zeros(48)
        tmp[ind] = 1
        encodings.append(list(tmp))

    signs -= np.mean(signs)

    pred = model.predict(signs)
    pad = np.zeros(48)
    groups = []
    back = ""
    count = 0
    for i in labels:
        if i != back:
            count += 1
        groups.append(count)
        back = i
    print(np.max(groups))
    maxv = count
    curr = 1
    X = []
    y = []
    while curr <= count:
        tmp = []
        size = groups.count(curr)
        if size < time_steps and size != 1:
            zeros = time_steps - size
            ind = groups.index(curr)
            for i in range(zeros):
                tmp.append(pad)
            for i in range(ind, ind + size):
                tmp.append(pred[i])
        elif size > time_steps:
            red = size - time_steps
            ind = groups.index(curr)
            for i in range(ind + red, ind + red + time_steps):
                tmp.append(pred[i])
        elif size != 1:
            ind = groups.index(curr)
            for i in range(ind, ind + time_steps-1):
                tmp.append(pred[i])
        if size != 1:
            X.append(tmp)
            y.append(encodings[groups.index(curr)])
        curr += 1
    return np.array(X), np.array(y)

