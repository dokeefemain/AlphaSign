import matplotlib.pyplot as plt
import numpy as np
def plot_bb(history):
    val = [max(history.history['val_bounding_box_accuracy']),np.argmax(history.history['val_bounding_box_accuracy'])]
    train = [max(history.history['bounding_box_accuracy']),np.argmax(history.history['bounding_box_accuracy'])]

    x_ax = list(np.linspace(0, len(history.history['bounding_box_accuracy']), num=9))
    x_ax.append(val[1])
    y_ax = list(np.linspace(0, 1, num=6))
    y_ax.append(val[0])


    plt.figure(0)

    plt.plot(history['bounding_box_accuracy'], label='class bounding box accuracy')
    plt.plot(history['val_bounding_box_accuracy'], label='val bounding box accuracy')

    plt.plot(train[1], train[0], 'bo')
    plt.plot([train[1],train[1]],[0, train[0]],':b')
    plt.plot([0,train[1]],[train[0], train[0]],':b')

    plt.plot(val[1],val[0], 'ro')
    plt.plot([val[1],val[1]],[0, val[0]],':r')
    plt.plot([0,val[1]],[val[0], val[0]],':r')

    plt.gcf().set_size_inches(14,7)
    plt.title('Bounding Box Accuracy')
    plt.xlabel('epochs')
    plt.xticks(x_ax)
    plt.ylabel('accuracy')
    plt.yticks(y_ax)
    plt.legend()

def plot(history):
    val = [max(history.history['val_accuracy']),np.argmax(history.history['val_accuracy'])]
    train = [max(history.history['accuracy']),np.argmax(history.history['accuracy'])]

    x_ax = list(np.linspace(0, len(history.history['accuracy']), num=9))
    x_ax.append(val[1])
    y_ax = list(np.linspace(0, 1, num=6))
    y_ax.append(val[0])


    plt.figure(0)

    plt.plot(history.history['accuracy'], label='class accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')

    plt.plot(train[1], train[0], 'bo')
    plt.plot([train[1],train[1]],[0, train[0]],':b')
    plt.plot([0,train[1]],[train[0], train[0]],':b')

    plt.plot(val[1],val[0], 'ro')
    plt.plot([val[1],val[1]],[0, val[0]],':r')
    plt.plot([0,val[1]],[val[0], val[0]],':r')

    plt.gcf().set_size_inches(14,7)
    plt.title('Label Accuracy')
    plt.xlabel('epochs')
    plt.xticks(x_ax)
    plt.ylabel('accuracy')
    plt.yticks(y_ax)
    plt.legend()