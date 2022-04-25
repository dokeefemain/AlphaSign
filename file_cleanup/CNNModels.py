{
 "cells": [
  {
   "cell_type": "markdown",
   "source": [
    "# CNN"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "import cv2\n",
    "import tensorflow as tf"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "outputs": [
    {
     "data": {
      "text/plain": "                                               Filename   Annotation tag  \\\n0     aiua120214-0/frameAnnotations-DataLog02142012_...             stop   \n1     aiua120214-0/frameAnnotations-DataLog02142012_...  speedLimitUrdbl   \n2     aiua120214-0/frameAnnotations-DataLog02142012_...             stop   \n3     aiua120214-0/frameAnnotations-DataLog02142012_...     speedLimit25   \n4     aiua120214-0/frameAnnotations-DataLog02142012_...     speedLimit25   \n...                                                 ...              ...   \n7850  vid9/frameAnnotations-MVI_0121.MOV_annotations...     speedLimit35   \n7851  vid9/frameAnnotations-MVI_0121.MOV_annotations...     speedLimit35   \n7852  vid9/frameAnnotations-MVI_0121.MOV_annotations...     speedLimit35   \n7853  vid9/frameAnnotations-MVI_0121.MOV_annotations...     speedLimit35   \n7854  vid9/frameAnnotations-MVI_0121.MOV_annotations...     speedLimit35   \n\n      Upper left corner X  Upper left corner Y  Lower right corner X  \\\n0                     862                  104                   916   \n1                     425                  197                   438   \n2                     922                   88                   982   \n3                     447                  193                   461   \n4                     469                  189                   483   \n...                   ...                  ...                   ...   \n7850                   41                  209                    65   \n7851                  526                  213                   543   \n7852                  546                  208                   564   \n7853                  573                  204                   592   \n7854                  604                  196                   628   \n\n      Lower right corner Y Occluded,On another road  \\\n0                      158                      0,0   \n1                      213                      0,0   \n2                      148                      1,0   \n3                      210                      0,0   \n4                      207                      0,0   \n...                    ...                      ...   \n7850                   239                      0,0   \n7851                   233                      0,0   \n7852                   230                      0,0   \n7853                   228                      0,0   \n7854                   223                      0,0   \n\n                                           Origin file  Origin frame number  \\\n0     aiua120214-0/DataLog02142012_external_camera.avi                 2667   \n1     aiua120214-0/DataLog02142012_external_camera.avi                 2667   \n2     aiua120214-0/DataLog02142012_external_camera.avi                 2672   \n3     aiua120214-0/DataLog02142012_external_camera.avi                 2672   \n4     aiua120214-0/DataLog02142012_external_camera.avi                 2677   \n...                                                ...                  ...   \n7850                                 vid9/MVI_0121.MOV                 8813   \n7851                                 vid9/MVI_0121.MOV                 8875   \n7852                                 vid9/MVI_0121.MOV                 8880   \n7853                                 vid9/MVI_0121.MOV                 8885   \n7854                                 vid9/MVI_0121.MOV                 8890   \n\n                   Origin track  Origin track frame number  \n0           stop_1330545910.avi                          2  \n1           stop_1330545910.avi                          2  \n2           stop_1330545910.avi                          7  \n3           stop_1330545910.avi                          7  \n4           stop_1330545910.avi                         12  \n...                         ...                        ...  \n7850  speedLimit_1324866802.avi                         22  \n7851  speedLimit_1324866807.avi                          2  \n7852  speedLimit_1324866807.avi                          7  \n7853  speedLimit_1324866807.avi                         12  \n7854  speedLimit_1324866807.avi                         17  \n\n[7855 rows x 11 columns]",
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Filename</th>\n      <th>Annotation tag</th>\n      <th>Upper left corner X</th>\n      <th>Upper left corner Y</th>\n      <th>Lower right corner X</th>\n      <th>Lower right corner Y</th>\n      <th>Occluded,On another road</th>\n      <th>Origin file</th>\n      <th>Origin frame number</th>\n      <th>Origin track</th>\n      <th>Origin track frame number</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>aiua120214-0/frameAnnotations-DataLog02142012_...</td>\n      <td>stop</td>\n      <td>862</td>\n      <td>104</td>\n      <td>916</td>\n      <td>158</td>\n      <td>0,0</td>\n      <td>aiua120214-0/DataLog02142012_external_camera.avi</td>\n      <td>2667</td>\n      <td>stop_1330545910.avi</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>aiua120214-0/frameAnnotations-DataLog02142012_...</td>\n      <td>speedLimitUrdbl</td>\n      <td>425</td>\n      <td>197</td>\n      <td>438</td>\n      <td>213</td>\n      <td>0,0</td>\n      <td>aiua120214-0/DataLog02142012_external_camera.avi</td>\n      <td>2667</td>\n      <td>stop_1330545910.avi</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>aiua120214-0/frameAnnotations-DataLog02142012_...</td>\n      <td>stop</td>\n      <td>922</td>\n      <td>88</td>\n      <td>982</td>\n      <td>148</td>\n      <td>1,0</td>\n      <td>aiua120214-0/DataLog02142012_external_camera.avi</td>\n      <td>2672</td>\n      <td>stop_1330545910.avi</td>\n      <td>7</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>aiua120214-0/frameAnnotations-DataLog02142012_...</td>\n      <td>speedLimit25</td>\n      <td>447</td>\n      <td>193</td>\n      <td>461</td>\n      <td>210</td>\n      <td>0,0</td>\n      <td>aiua120214-0/DataLog02142012_external_camera.avi</td>\n      <td>2672</td>\n      <td>stop_1330545910.avi</td>\n      <td>7</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>aiua120214-0/frameAnnotations-DataLog02142012_...</td>\n      <td>speedLimit25</td>\n      <td>469</td>\n      <td>189</td>\n      <td>483</td>\n      <td>207</td>\n      <td>0,0</td>\n      <td>aiua120214-0/DataLog02142012_external_camera.avi</td>\n      <td>2677</td>\n      <td>stop_1330545910.avi</td>\n      <td>12</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>7850</th>\n      <td>vid9/frameAnnotations-MVI_0121.MOV_annotations...</td>\n      <td>speedLimit35</td>\n      <td>41</td>\n      <td>209</td>\n      <td>65</td>\n      <td>239</td>\n      <td>0,0</td>\n      <td>vid9/MVI_0121.MOV</td>\n      <td>8813</td>\n      <td>speedLimit_1324866802.avi</td>\n      <td>22</td>\n    </tr>\n    <tr>\n      <th>7851</th>\n      <td>vid9/frameAnnotations-MVI_0121.MOV_annotations...</td>\n      <td>speedLimit35</td>\n      <td>526</td>\n      <td>213</td>\n      <td>543</td>\n      <td>233</td>\n      <td>0,0</td>\n      <td>vid9/MVI_0121.MOV</td>\n      <td>8875</td>\n      <td>speedLimit_1324866807.avi</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>7852</th>\n      <td>vid9/frameAnnotations-MVI_0121.MOV_annotations...</td>\n      <td>speedLimit35</td>\n      <td>546</td>\n      <td>208</td>\n      <td>564</td>\n      <td>230</td>\n      <td>0,0</td>\n      <td>vid9/MVI_0121.MOV</td>\n      <td>8880</td>\n      <td>speedLimit_1324866807.avi</td>\n      <td>7</td>\n    </tr>\n    <tr>\n      <th>7853</th>\n      <td>vid9/frameAnnotations-MVI_0121.MOV_annotations...</td>\n      <td>speedLimit35</td>\n      <td>573</td>\n      <td>204</td>\n      <td>592</td>\n      <td>228</td>\n      <td>0,0</td>\n      <td>vid9/MVI_0121.MOV</td>\n      <td>8885</td>\n      <td>speedLimit_1324866807.avi</td>\n      <td>12</td>\n    </tr>\n    <tr>\n      <th>7854</th>\n      <td>vid9/frameAnnotations-MVI_0121.MOV_annotations...</td>\n      <td>speedLimit35</td>\n      <td>604</td>\n      <td>196</td>\n      <td>628</td>\n      <td>223</td>\n      <td>0,0</td>\n      <td>vid9/MVI_0121.MOV</td>\n      <td>8890</td>\n      <td>speedLimit_1324866807.avi</td>\n      <td>17</td>\n    </tr>\n  </tbody>\n</table>\n<p>7855 rows × 11 columns</p>\n</div>"
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_ann = pd.read_csv(\"lib/datasets/LISA/allAnnotations.csv\", delimiter=';')\n",
    "all_ann"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "outputs": [],
   "source": [
    "#max image size\n",
    "# gt 260 400\n",
    "height = 50\n",
    "width = 50\n",
    "data = []\n",
    "labels = []\n",
    "path = \"lib/datasets/LISA/\"\n",
    "test = 0\n",
    "for i in range(len(all_ann[\"Filename\"])):\n",
    "    file = all_ann[\"Filename\"][i]\n",
    "    sign = all_ann[\"Annotation tag\"][i]\n",
    "    image = Image.open(path+file)\n",
    "    image = image.resize((width,height))\n",
    "    image = np.asarray(image)\n",
    "    data.append(image)\n",
    "    labels.append(sign)\n",
    "\n",
    "\n",
    "signs = np.array(data)\n",
    "labels = np.array(labels)\n",
    "\n",
    "# Randomize order\n",
    "s = np.arange(signs.shape[0])\n",
    "np.random.seed(43)\n",
    "np.random.shuffle(s)\n",
    "signs = signs[s]\n",
    "labels = labels[s]"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "outputs": [],
   "source": [
    "def one_hot(lab, vals):\n",
    "    hot = []\n",
    "    for i in range(len(vals)):\n",
    "        # tmp = np.zeros(len(lab))\n",
    "        # tmp[np.where(lab == i)] = 1\n",
    "        hot.append(lab.tolist().index(vals[i]))\n",
    "        # hot[i] = np.where(lab == vals[i])\n",
    "        # hot.append(tmp)\n",
    "    return hot"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from imblearn.over_sampling import  RandomOverSampler\n",
    "X_train, _, y_train, _ = train_test_split(signs, labels, test_size=0.2, random_state=1)\n",
    "del signs\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "outputs": [],
   "source": [
    "# Preprocess\n",
    "X_train = X_train.astype('float64')\n",
    "X_val = X_val.astype('float64')\n",
    "#X_test = X_test.astype('float64')\n",
    "\n",
    "# Normalize\n",
    "mean_image = np.mean(X_train, axis = 0)\n",
    "X_train -= mean_image\n",
    "X_val -= mean_image\n",
    "#X_test -= mean_image"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "outputs": [],
   "source": [
    "# Encode\n",
    "lab = np.unique(labels)\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "y_train_e = to_categorical(one_hot(lab,y_train),47)\n",
    "#y_test_e = to_categorical(one_hot(lab,y_test),47)\n",
    "y_val_e = to_categorical(one_hot(lab,y_val),47)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "outputs": [
    {
     "data": {
      "text/plain": "(4713, 260, 400, 3)"
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## CNN models"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, ZeroPadding2D"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### DevinNet"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/40\n",
      "148/148 [==============================] - 8s 20ms/step - loss: 8.7624 - accuracy: 0.2058 - val_loss: 2.8330 - val_accuracy: 0.2693\n",
      "Epoch 2/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 2.6274 - accuracy: 0.2898 - val_loss: 2.1269 - val_accuracy: 0.4137\n",
      "Epoch 3/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 2.0189 - accuracy: 0.4188 - val_loss: 1.6253 - val_accuracy: 0.5614\n",
      "Epoch 4/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 1.5874 - accuracy: 0.5302 - val_loss: 1.2430 - val_accuracy: 0.6556\n",
      "Epoch 5/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 1.2499 - accuracy: 0.6155 - val_loss: 0.9992 - val_accuracy: 0.7015\n",
      "Epoch 6/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.9921 - accuracy: 0.6955 - val_loss: 0.9233 - val_accuracy: 0.7498\n",
      "Epoch 7/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.8635 - accuracy: 0.7320 - val_loss: 0.8106 - val_accuracy: 0.7696\n",
      "Epoch 8/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.7042 - accuracy: 0.7787 - val_loss: 0.6995 - val_accuracy: 0.7989\n",
      "Epoch 9/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.6248 - accuracy: 0.8016 - val_loss: 0.6252 - val_accuracy: 0.8014\n",
      "Epoch 10/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.5397 - accuracy: 0.8194 - val_loss: 0.5994 - val_accuracy: 0.8269\n",
      "Epoch 11/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.4977 - accuracy: 0.8375 - val_loss: 0.5883 - val_accuracy: 0.8243\n",
      "Epoch 12/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.4610 - accuracy: 0.8455 - val_loss: 0.5855 - val_accuracy: 0.8218\n",
      "Epoch 13/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.4303 - accuracy: 0.8578 - val_loss: 0.5621 - val_accuracy: 0.8250\n",
      "Epoch 14/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.4384 - accuracy: 0.8500 - val_loss: 0.5204 - val_accuracy: 0.8275\n",
      "Epoch 15/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.3936 - accuracy: 0.8706 - val_loss: 0.5099 - val_accuracy: 0.8332\n",
      "Epoch 16/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.3747 - accuracy: 0.8706 - val_loss: 0.5315 - val_accuracy: 0.8275\n",
      "Epoch 17/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.3454 - accuracy: 0.8833 - val_loss: 0.4985 - val_accuracy: 0.8377\n",
      "Epoch 18/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.3276 - accuracy: 0.8837 - val_loss: 0.5158 - val_accuracy: 0.8320\n",
      "Epoch 19/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.3330 - accuracy: 0.8784 - val_loss: 0.5182 - val_accuracy: 0.8364\n",
      "Epoch 20/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.3513 - accuracy: 0.8791 - val_loss: 0.4928 - val_accuracy: 0.8364\n",
      "Epoch 21/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.3110 - accuracy: 0.8916 - val_loss: 0.4701 - val_accuracy: 0.8466\n",
      "Epoch 22/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.3067 - accuracy: 0.8922 - val_loss: 0.4769 - val_accuracy: 0.8339\n",
      "Epoch 23/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.3060 - accuracy: 0.8922 - val_loss: 0.4982 - val_accuracy: 0.8332\n",
      "Epoch 24/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2786 - accuracy: 0.8999 - val_loss: 0.5166 - val_accuracy: 0.8351\n",
      "Epoch 25/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2850 - accuracy: 0.8928 - val_loss: 0.4729 - val_accuracy: 0.8511\n",
      "Epoch 26/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.2937 - accuracy: 0.8928 - val_loss: 0.5245 - val_accuracy: 0.8434\n",
      "Epoch 27/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2790 - accuracy: 0.8971 - val_loss: 0.5256 - val_accuracy: 0.8415\n",
      "Epoch 28/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.2805 - accuracy: 0.8931 - val_loss: 0.4750 - val_accuracy: 0.8421\n",
      "Epoch 29/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.2823 - accuracy: 0.9009 - val_loss: 0.4909 - val_accuracy: 0.8415\n",
      "Epoch 30/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2682 - accuracy: 0.9003 - val_loss: 0.4958 - val_accuracy: 0.8447\n",
      "Epoch 31/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2575 - accuracy: 0.9052 - val_loss: 0.5451 - val_accuracy: 0.8428\n",
      "Epoch 32/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2652 - accuracy: 0.8990 - val_loss: 0.5016 - val_accuracy: 0.8377\n",
      "Epoch 33/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2619 - accuracy: 0.9043 - val_loss: 0.4764 - val_accuracy: 0.8447\n",
      "Epoch 34/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2584 - accuracy: 0.9022 - val_loss: 0.5198 - val_accuracy: 0.8402\n",
      "Epoch 35/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.2504 - accuracy: 0.9081 - val_loss: 0.5367 - val_accuracy: 0.8511\n",
      "Epoch 36/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.2481 - accuracy: 0.9030 - val_loss: 0.4967 - val_accuracy: 0.8434\n",
      "Epoch 37/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2551 - accuracy: 0.9009 - val_loss: 0.5336 - val_accuracy: 0.8415\n",
      "Epoch 38/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2452 - accuracy: 0.9045 - val_loss: 0.5102 - val_accuracy: 0.8479\n",
      "Epoch 39/40\n",
      "148/148 [==============================] - 2s 16ms/step - loss: 0.2420 - accuracy: 0.9086 - val_loss: 0.5211 - val_accuracy: 0.8491\n",
      "Epoch 40/40\n",
      "148/148 [==============================] - 2s 17ms/step - loss: 0.2456 - accuracy: 0.9060 - val_loss: 0.5163 - val_accuracy: 0.8504\n"
     ]
    }
   ],
   "source": [
    "model=Sequential()\n",
    "#adding convolution layer\n",
    "#model.add(ZeroPadding2D(padding=(2,2)))\n",
    "model.add(Conv2D(96,(6,4),strides=4,activation='relu',input_shape=X_train.shape[1:]))\n",
    "model.add(MaxPool2D((3,3)))\n",
    "model.add(Dropout(rate=0.25))\n",
    "\n",
    "model.add(Conv2D(64,(3,3),strides=1,activation='relu'))\n",
    "model.add(MaxPool2D((2,2)))\n",
    "model.add(Dropout(rate=0.25))\n",
    "\n",
    "model.add(Conv2D(64,(3,3),strides=1,activation='relu'))\n",
    "model.add(MaxPool2D((2,2)))\n",
    "model.add(Dropout(rate=0.25))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(256,activation=\"relu\"))\n",
    "model.add(Dropout(rate=0.25))\n",
    "model.add(Dense(len(lab),activation=\"softmax\"))\n",
    "\n",
    "model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])\n",
    "epochs = 17\n",
    "history = model.fit(X_train, y_train_e, batch_size=32, epochs=epochs,\n",
    "validation_data=(X_val, y_val_e))"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### AlexNet"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/17\n",
      "148/148 [==============================] - 13s 52ms/step - loss: 7.6359 - accuracy: 0.2205 - val_loss: 3.1082 - val_accuracy: 0.1846\n",
      "Epoch 2/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 2.7214 - accuracy: 0.2389 - val_loss: 2.5870 - val_accuracy: 0.2553\n",
      "Epoch 3/17\n",
      "148/148 [==============================] - 5s 31ms/step - loss: 2.5284 - accuracy: 0.2924 - val_loss: 2.3541 - val_accuracy: 0.3113\n",
      "Epoch 4/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 2.1445 - accuracy: 0.3821 - val_loss: 1.9926 - val_accuracy: 0.4067\n",
      "Epoch 5/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 1.7273 - accuracy: 0.4897 - val_loss: 1.7091 - val_accuracy: 0.4990\n",
      "Epoch 6/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 1.3581 - accuracy: 0.5922 - val_loss: 1.1739 - val_accuracy: 0.6467\n",
      "Epoch 7/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.9179 - accuracy: 0.7138 - val_loss: 1.0818 - val_accuracy: 0.6856\n",
      "Epoch 8/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.6753 - accuracy: 0.7846 - val_loss: 0.9082 - val_accuracy: 0.7358\n",
      "Epoch 9/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.5415 - accuracy: 0.8273 - val_loss: 0.7594 - val_accuracy: 0.7906\n",
      "Epoch 10/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.4726 - accuracy: 0.8449 - val_loss: 0.7652 - val_accuracy: 0.7778\n",
      "Epoch 11/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.3754 - accuracy: 0.8731 - val_loss: 0.6628 - val_accuracy: 0.8256\n",
      "Epoch 12/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.3863 - accuracy: 0.8661 - val_loss: 0.8068 - val_accuracy: 0.7836\n",
      "Epoch 13/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.3863 - accuracy: 0.8646 - val_loss: 0.7802 - val_accuracy: 0.7810\n",
      "Epoch 14/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.3428 - accuracy: 0.8750 - val_loss: 0.6469 - val_accuracy: 0.8071\n",
      "Epoch 15/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.2973 - accuracy: 0.8865 - val_loss: 0.7404 - val_accuracy: 0.8033\n",
      "Epoch 16/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.2984 - accuracy: 0.8854 - val_loss: 0.6809 - val_accuracy: 0.8008\n",
      "Epoch 17/17\n",
      "148/148 [==============================] - 4s 30ms/step - loss: 0.3532 - accuracy: 0.8723 - val_loss: 0.7212 - val_accuracy: 0.8084\n"
     ]
    }
   ],
   "source": [
    "\n",
    "model=Sequential()\n",
    "#adding convolution layer\n",
    "\n",
    "model.add(Conv2D(64,(11,11),strides=4,activation='relu',input_shape=X_train.shape[1:],padding='valid'))\n",
    "model.add(MaxPool2D((3,3),strides=2))\n",
    "\n",
    "model.add(Conv2D(192,(5,5),activation='relu'))\n",
    "model.add(MaxPool2D((3,3), strides=2))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=1))\n",
    "model.add(Conv2D(384,(3,3),activation='relu',padding='valid'))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=1))\n",
    "model.add(Conv2D(384,(3,3),activation='relu',padding='valid'))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=1))\n",
    "model.add(Conv2D(256,(3,3),activation='relu',padding='valid'))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(4096,activation=\"relu\"))\n",
    "model.add(Dense(4096,activation=\"relu\"))\n",
    "model.add(Dense(len(lab),activation=\"softmax\"))\n",
    "\n",
    "model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])\n",
    "epochs = 17\n",
    "history = model.fit(X_train, y_train_e, batch_size=32, epochs=epochs,\n",
    "validation_data=(X_val, y_val_e))"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Another Version of AlexNet"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/17\n",
      "148/148 [==============================] - 6s 38ms/step - loss: 12.9129 - accuracy: 0.2194 - val_loss: 2.8564 - val_accuracy: 0.2177\n",
      "Epoch 2/17\n",
      "148/148 [==============================] - 4s 27ms/step - loss: 2.8606 - accuracy: 0.2311 - val_loss: 2.8415 - val_accuracy: 0.2177\n",
      "Epoch 3/17\n",
      "148/148 [==============================] - 4s 26ms/step - loss: 2.8627 - accuracy: 0.2311 - val_loss: 2.8412 - val_accuracy: 0.2177\n",
      "Epoch 4/17\n",
      " 16/148 [==>...........................] - ETA: 3s - loss: 2.7482 - accuracy: 0.2676"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mKeyboardInterrupt\u001B[0m                         Traceback (most recent call last)",
      "\u001B[1;32m~\\AppData\\Local\\Temp/ipykernel_5616/2488044188.py\u001B[0m in \u001B[0;36m<module>\u001B[1;34m\u001B[0m\n\u001B[0;32m     31\u001B[0m \u001B[0mepochs\u001B[0m \u001B[1;33m=\u001B[0m \u001B[1;36m17\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m     32\u001B[0m history = model.fit(X_train, y_train_e, batch_size=32, epochs=epochs,\n\u001B[1;32m---> 33\u001B[1;33m validation_data=(X_val, y_val_e))\n\u001B[0m",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\utils\\traceback_utils.py\u001B[0m in \u001B[0;36merror_handler\u001B[1;34m(*args, **kwargs)\u001B[0m\n\u001B[0;32m     62\u001B[0m     \u001B[0mfiltered_tb\u001B[0m \u001B[1;33m=\u001B[0m \u001B[1;32mNone\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m     63\u001B[0m     \u001B[1;32mtry\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m---> 64\u001B[1;33m       \u001B[1;32mreturn\u001B[0m \u001B[0mfn\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m*\u001B[0m\u001B[0margs\u001B[0m\u001B[1;33m,\u001B[0m \u001B[1;33m**\u001B[0m\u001B[0mkwargs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m     65\u001B[0m     \u001B[1;32mexcept\u001B[0m \u001B[0mException\u001B[0m \u001B[1;32mas\u001B[0m \u001B[0me\u001B[0m\u001B[1;33m:\u001B[0m  \u001B[1;31m# pylint: disable=broad-except\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m     66\u001B[0m       \u001B[0mfiltered_tb\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0m_process_traceback_frames\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0me\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m__traceback__\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\engine\\training.py\u001B[0m in \u001B[0;36mfit\u001B[1;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)\u001B[0m\n\u001B[0;32m   1387\u001B[0m               \u001B[0mlogs\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mtmp_logs\u001B[0m  \u001B[1;31m# No error, now safe to assign to logs.\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1388\u001B[0m               \u001B[0mend_step\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mstep\u001B[0m \u001B[1;33m+\u001B[0m \u001B[0mdata_handler\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mstep_increment\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m-> 1389\u001B[1;33m               \u001B[0mcallbacks\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mon_train_batch_end\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mend_step\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m   1390\u001B[0m               \u001B[1;32mif\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mstop_training\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1391\u001B[0m                 \u001B[1;32mbreak\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\callbacks.py\u001B[0m in \u001B[0;36mon_train_batch_end\u001B[1;34m(self, batch, logs)\u001B[0m\n\u001B[0;32m    436\u001B[0m     \"\"\"\n\u001B[0;32m    437\u001B[0m     \u001B[1;32mif\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_should_call_train_batch_hooks\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m--> 438\u001B[1;33m       \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_call_batch_hook\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mModeKeys\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mTRAIN\u001B[0m\u001B[1;33m,\u001B[0m \u001B[1;34m'end'\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m=\u001B[0m\u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    439\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    440\u001B[0m   \u001B[1;32mdef\u001B[0m \u001B[0mon_test_batch_begin\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mself\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m=\u001B[0m\u001B[1;32mNone\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\callbacks.py\u001B[0m in \u001B[0;36m_call_batch_hook\u001B[1;34m(self, mode, hook, batch, logs)\u001B[0m\n\u001B[0;32m    295\u001B[0m       \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_call_batch_begin_hook\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mmode\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    296\u001B[0m     \u001B[1;32melif\u001B[0m \u001B[0mhook\u001B[0m \u001B[1;33m==\u001B[0m \u001B[1;34m'end'\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m--> 297\u001B[1;33m       \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_call_batch_end_hook\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mmode\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    298\u001B[0m     \u001B[1;32melse\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    299\u001B[0m       raise ValueError(\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\callbacks.py\u001B[0m in \u001B[0;36m_call_batch_end_hook\u001B[1;34m(self, mode, batch, logs)\u001B[0m\n\u001B[0;32m    316\u001B[0m       \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_batch_times\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mappend\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mbatch_time\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    317\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m--> 318\u001B[1;33m     \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_call_batch_hook_helper\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mhook_name\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    319\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    320\u001B[0m     \u001B[1;32mif\u001B[0m \u001B[0mlen\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_batch_times\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;33m>=\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_num_batches_for_timing_check\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\callbacks.py\u001B[0m in \u001B[0;36m_call_batch_hook_helper\u001B[1;34m(self, hook_name, batch, logs)\u001B[0m\n\u001B[0;32m    354\u001B[0m     \u001B[1;32mfor\u001B[0m \u001B[0mcallback\u001B[0m \u001B[1;32min\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mcallbacks\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    355\u001B[0m       \u001B[0mhook\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mgetattr\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mcallback\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mhook_name\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m--> 356\u001B[1;33m       \u001B[0mhook\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    357\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    358\u001B[0m     \u001B[1;32mif\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_check_timing\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\callbacks.py\u001B[0m in \u001B[0;36mon_train_batch_end\u001B[1;34m(self, batch, logs)\u001B[0m\n\u001B[0;32m   1032\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1033\u001B[0m   \u001B[1;32mdef\u001B[0m \u001B[0mon_train_batch_end\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mself\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m=\u001B[0m\u001B[1;32mNone\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m-> 1034\u001B[1;33m     \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_batch_update_progbar\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m   1035\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1036\u001B[0m   \u001B[1;32mdef\u001B[0m \u001B[0mon_test_batch_end\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mself\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mbatch\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlogs\u001B[0m\u001B[1;33m=\u001B[0m\u001B[1;32mNone\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\callbacks.py\u001B[0m in \u001B[0;36m_batch_update_progbar\u001B[1;34m(self, batch, logs)\u001B[0m\n\u001B[0;32m   1104\u001B[0m     \u001B[1;32mif\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mverbose\u001B[0m \u001B[1;33m==\u001B[0m \u001B[1;36m1\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1105\u001B[0m       \u001B[1;31m# Only block async when verbose = 1.\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m-> 1106\u001B[1;33m       \u001B[0mlogs\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mtf_utils\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0msync_to_numpy_or_python_type\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mlogs\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m   1107\u001B[0m       \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mprogbar\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mupdate\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mseen\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mlist\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mlogs\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mitems\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mfinalize\u001B[0m\u001B[1;33m=\u001B[0m\u001B[1;32mFalse\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1108\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\utils\\tf_utils.py\u001B[0m in \u001B[0;36msync_to_numpy_or_python_type\u001B[1;34m(tensors)\u001B[0m\n\u001B[0;32m    561\u001B[0m     \u001B[1;32mreturn\u001B[0m \u001B[0mt\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mitem\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;32mif\u001B[0m \u001B[0mnp\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mndim\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mt\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;33m==\u001B[0m \u001B[1;36m0\u001B[0m \u001B[1;32melse\u001B[0m \u001B[0mt\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    562\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m--> 563\u001B[1;33m   \u001B[1;32mreturn\u001B[0m \u001B[0mtf\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mnest\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mmap_structure\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0m_to_single_numpy_or_python_type\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mtensors\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    564\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    565\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\tensorflow\\python\\util\\nest.py\u001B[0m in \u001B[0;36mmap_structure\u001B[1;34m(func, *structure, **kwargs)\u001B[0m\n\u001B[0;32m    912\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    913\u001B[0m   return pack_sequence_as(\n\u001B[1;32m--> 914\u001B[1;33m       \u001B[0mstructure\u001B[0m\u001B[1;33m[\u001B[0m\u001B[1;36m0\u001B[0m\u001B[1;33m]\u001B[0m\u001B[1;33m,\u001B[0m \u001B[1;33m[\u001B[0m\u001B[0mfunc\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m*\u001B[0m\u001B[0mx\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;32mfor\u001B[0m \u001B[0mx\u001B[0m \u001B[1;32min\u001B[0m \u001B[0mentries\u001B[0m\u001B[1;33m]\u001B[0m\u001B[1;33m,\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    915\u001B[0m       expand_composites=expand_composites)\n\u001B[0;32m    916\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\tensorflow\\python\\util\\nest.py\u001B[0m in \u001B[0;36m<listcomp>\u001B[1;34m(.0)\u001B[0m\n\u001B[0;32m    912\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    913\u001B[0m   return pack_sequence_as(\n\u001B[1;32m--> 914\u001B[1;33m       \u001B[0mstructure\u001B[0m\u001B[1;33m[\u001B[0m\u001B[1;36m0\u001B[0m\u001B[1;33m]\u001B[0m\u001B[1;33m,\u001B[0m \u001B[1;33m[\u001B[0m\u001B[0mfunc\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m*\u001B[0m\u001B[0mx\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;32mfor\u001B[0m \u001B[0mx\u001B[0m \u001B[1;32min\u001B[0m \u001B[0mentries\u001B[0m\u001B[1;33m]\u001B[0m\u001B[1;33m,\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    915\u001B[0m       expand_composites=expand_composites)\n\u001B[0;32m    916\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\keras\\utils\\tf_utils.py\u001B[0m in \u001B[0;36m_to_single_numpy_or_python_type\u001B[1;34m(t)\u001B[0m\n\u001B[0;32m    555\u001B[0m     \u001B[1;31m# Don't turn ragged or sparse tensors to NumPy.\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    556\u001B[0m     \u001B[1;32mif\u001B[0m \u001B[0misinstance\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mt\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mtf\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mTensor\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m--> 557\u001B[1;33m       \u001B[0mt\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mt\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mnumpy\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m    558\u001B[0m     \u001B[1;31m# Strings, ragged and sparse tensors don't have .item(). Return them as-is.\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m    559\u001B[0m     \u001B[1;32mif\u001B[0m \u001B[1;32mnot\u001B[0m \u001B[0misinstance\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mt\u001B[0m\u001B[1;33m,\u001B[0m \u001B[1;33m(\u001B[0m\u001B[0mnp\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mndarray\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mnp\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mgeneric\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\tensorflow\\python\\framework\\ops.py\u001B[0m in \u001B[0;36mnumpy\u001B[1;34m(self)\u001B[0m\n\u001B[0;32m   1221\u001B[0m     \"\"\"\n\u001B[0;32m   1222\u001B[0m     \u001B[1;31m# TODO(slebedev): Consider avoiding a copy for non-CPU or remote tensors.\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m-> 1223\u001B[1;33m     \u001B[0mmaybe_arr\u001B[0m \u001B[1;33m=\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_numpy\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m)\u001B[0m  \u001B[1;31m# pylint: disable=protected-access\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m   1224\u001B[0m     \u001B[1;32mreturn\u001B[0m \u001B[0mmaybe_arr\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mcopy\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;32mif\u001B[0m \u001B[0misinstance\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mmaybe_arr\u001B[0m\u001B[1;33m,\u001B[0m \u001B[0mnp\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0mndarray\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;32melse\u001B[0m \u001B[0mmaybe_arr\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1225\u001B[0m \u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;32m~\\anaconda3\\envs\\tensor\\lib\\site-packages\\tensorflow\\python\\framework\\ops.py\u001B[0m in \u001B[0;36m_numpy\u001B[1;34m(self)\u001B[0m\n\u001B[0;32m   1187\u001B[0m   \u001B[1;32mdef\u001B[0m \u001B[0m_numpy\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0mself\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1188\u001B[0m     \u001B[1;32mtry\u001B[0m\u001B[1;33m:\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[1;32m-> 1189\u001B[1;33m       \u001B[1;32mreturn\u001B[0m \u001B[0mself\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_numpy_internal\u001B[0m\u001B[1;33m(\u001B[0m\u001B[1;33m)\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0m\u001B[0;32m   1190\u001B[0m     \u001B[1;32mexcept\u001B[0m \u001B[0mcore\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_NotOkStatusException\u001B[0m \u001B[1;32mas\u001B[0m \u001B[0me\u001B[0m\u001B[1;33m:\u001B[0m  \u001B[1;31m# pylint: disable=protected-access\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n\u001B[0;32m   1191\u001B[0m       \u001B[1;32mraise\u001B[0m \u001B[0mcore\u001B[0m\u001B[1;33m.\u001B[0m\u001B[0m_status_to_exception\u001B[0m\u001B[1;33m(\u001B[0m\u001B[0me\u001B[0m\u001B[1;33m)\u001B[0m \u001B[1;32mfrom\u001B[0m \u001B[1;32mNone\u001B[0m  \u001B[1;31m# pylint: disable=protected-access\u001B[0m\u001B[1;33m\u001B[0m\u001B[1;33m\u001B[0m\u001B[0m\n",
      "\u001B[1;31mKeyboardInterrupt\u001B[0m: "
     ]
    }
   ],
   "source": [
    "tf.keras.backend.clear_session()\n",
    "model=Sequential()\n",
    "#adding convolution layer\n",
    "\n",
    "model.add(Conv2D(96,(11,11),strides=4,activation='relu',input_shape=X_train.shape[1:]))\n",
    "model.add(MaxPool2D((3,3),strides=2))\n",
    "model.add(Dropout(rate=0.25))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=2))\n",
    "model.add(Conv2D(192,(5,5),strides=1,activation='relu',padding='valid'))\n",
    "model.add(MaxPool2D((3,3), strides=2))\n",
    "model.add(Dropout(rate=0.25))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=1))\n",
    "model.add(Conv2D(384,(3,3),strides=1,activation='relu',padding='valid'))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=1))\n",
    "model.add(Conv2D(384,(3,3),strides=1,activation='relu',padding='valid'))\n",
    "\n",
    "model.add(ZeroPadding2D(padding=1))\n",
    "model.add(Conv2D(256,(3,3),strides=1,activation='relu',padding='valid'))\n",
    "\n",
    "model.add(MaxPool2D((3,3), strides=2))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(4096,activation=\"relu\"))\n",
    "model.add(Dense(4096,activation=\"relu\"))\n",
    "model.add(Dense(len(lab),activation=\"softmax\"))\n",
    "\n",
    "model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])\n",
    "epochs = 17\n",
    "history = model.fit(X_train, y_train_e, batch_size=32, epochs=epochs,\n",
    "validation_data=(X_val, y_val_e))"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: App/lib/models/GermanModel/AlexNet\\assets\n"
     ]
    }
   ],
   "source": [
    "model.save(\"App/lib/models/GermanModel/AlexNet\")"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "val_accuracy: 0.7810\n",
    "3x3 3x3 val_accuracy: 0.7944\n",
    "5x5 3x3 e = 8 val_accuracy: 0.7976\n",
    "5x5 3x3 3x3 e = 8 val_accuracy: 0.8453\n",
    "5x5 3x3 3x3 d = 256 val_accuracy: 0.8504\n",
    "3x3 3x3 3x3 d = 256 val_accuracy: 0.8555\n",
    "3x3 pool 3x3 pool 3x3 pool val_accuracy: 0.8791\n",
    "3x3 pool 3x3 pool 3x3 3x3 pool val_accuracy: 0.8746"
   ],
   "metadata": {
    "collapsed": false
   }
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "outputs": [
    {
     "data": {
      "text/plain": "<matplotlib.legend.Legend at 0x17c406cd748>"
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": "<Figure size 432x288 with 1 Axes>",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": "<Figure size 432x288 with 1 Axes>",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(0)\n",
    "plt.plot(history.history['accuracy'], label='training accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='val accuracy')\n",
    "plt.title('Accuracy')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('accuracy')\n",
    "plt.legend()\n",
    "\n",
    "plt.figure(1)\n",
    "plt.plot(history.history['loss'], label='training loss')\n",
    "plt.plot(history.history['val_loss'], label='val loss')\n",
    "plt.title('Loss')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('loss')\n",
    "plt.legend()"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}