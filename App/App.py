# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:07:26 2021

@author: Minas Benyamin

@editor: Devin O'Keefe
"""

import time
import cv2
import mss
import numpy as np
import tensorflow as tf
import pandas as pd

def annotate_sign(image, ind,max_list, encoding):
    t = time.time()
    global start_time
    global curr_lab
    global curr_val
    prediction = encoding[encoding["Encoding"] == ind]["Label"].values[0]
    maxv = np.max(max_list)


    left_location = int(50 * image.shape[0] / 480)
    right_location = int(100 * image.shape[1] / 640)
    font_scale = max(image.shape) / 800

    if t-start_time > .5:

        idk = str(prediction) + " " + str(maxv)
        cv2.putText(image, idk, (left_location, right_location),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        start_time = time.time()

        curr_lab = prediction
        curr_val = ind
    else:
        idk = curr_lab + " " + str(max_list[0][ind])
        cv2.putText(image, idk, (left_location, right_location),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        print("test")


def checkPoseDetection(image, frame_window, encoding, model):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    global frame_count
    try:
        if(frame_count > frame_window):
            tmp = image.copy()

            img1 = cv2.resize(tmp, dsize=(227,227), interpolation=cv2.INTER_CUBIC)

            test_image = img1.astype('float64')

            test_image -= np.mean(test_image, axis=0)

            img1 = test_image.reshape((1, 227, 227, 3))
            tmp = model.predict(img1)
            ind = np.argmax(tmp)


            #0.9649314
            prediction = encoding[encoding["Encoding"] == ind]["Label"].values[0]
            image.flags.writeable = True

            annotate_sign(image, ind, tmp,encoding)
    except:
        image.flags.writeable = True
        left_location = int(50 * image.shape[0] / 480)
        font_scale = max(image.shape) / 800
        right_loc_1 = int(100 * image.shape[1] / 640)
        cv2.putText(image, "Looking for Target", (left_location, right_loc_1), cv2.FONT_HERSHEY_COMPLEX, font_scale,
                    (255, 50, 0), 2, lineType=cv2.LINE_AA)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    video_display(image)


def video_display(image):
    cv2.imshow('MediaPipe Pose', image)
    # cv2.imwrite('tmp/annotated_image' + str(counter) + '.png', image)
    vid_writer.write(image)


# For webcam input:
# cap = cv2.VideoCapture(input_source)
if __name__ == "__main__":

    input_source = 'garo_pullup_back.mov'
    output_source = 'output_36.mp4'
    start_time = time.time()
    curr_lab = ""
    curr_val = 0

    this_width = 1000
    this_height = 580
    monitor = {'top': 200, 'left': 0, 'width': this_width, 'height': this_height}

    # cap = cv2.VideoCapture(0)
    # hasFrame, image = cap.read()

    model = tf.keras.models.load_model("lib/Models/AlexNet")
    encoding = pd.read_csv("lib/datasets/key1.csv")
    vid_writer = cv2.VideoWriter(output_source, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                 (this_width, this_height))

    counter = 0
    condition = 1
    depth_condition = 1
    frame_count = 0
    bad_frames = 0

    var_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    old_x = var_x
    old_y = var_y
    frame_window = 10
    variance_collection = [0] * frame_window

    with mss.mss() as sct:
        while 'Screen capturing':
            frame_count = frame_count + 1
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            if True:
                checkPoseDetection(img, frame_window, encoding, model)
            else:
                print("Ignoring empty camera frame.")
                vid_writer.release()
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # hit escape to close video player
            if cv2.waitKey(5) & 0xFF == 27:
                vid_writer.release()
                break

        ratio = (bad_frames / frame_count) * 100
    print("Bad Frame Percentage: {:2.4}".format(ratio))
    if (ratio > 5):
        print('This video is not suitable for training the exercise prototypes at this time')

    cv2.destroyAllWindows()