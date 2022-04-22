from keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout, concatenate
from keras.models import Model
from keras import backend as K
import numpy as np

class IRV2():
    #299 299
    def __init__(self, input_shape, n_classes, activation, scale1, scale2, scale3, p=1, t=2, r=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3

    def build_model(self):
        input_data = Input(shape=self.input_shape)
        stem = self.stem(input_data)
        ira = Activation("relu")(stem)
        for i in range(5):
            ira = self.IRA(ira)

        ra = self.RA(ira)

        irb = Activation("relu")(ra)
        for i in range(10):
            irb = self.IRB(irb)

        rb = self.RB(irb)

        irc = Activation("relu")(rb)
        for i in range(5):
            irc = self.IRC(irc)

        pool = GlobalAveragePooling2D()(irc)

        drop = Dropout(0.2)(pool)
        final = Dense(self.n_classes, activation= self.activation)(drop)
        model = Model(inputs = input_data, outputs = final)

        return model


    def stem(self, input_data):
        conv_1 = Conv2D(32, (3,3), strides=2, padding="valid")(input_data)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation('relu')(conv_1)

        conv_2 = Conv2D(32, (3, 3), padding="valid")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation('relu')(conv_2)

        conv_3 = Conv2D(64, (3, 3), padding="same")(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation('relu')(conv_3)

        pool_1 = MaxPool2D((3,3), strides=2, padding='valid')(conv_3)

        conv_4 = Conv2D(96, (3, 3), strides = 2, padding="valid")(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation('relu')(conv_4)

        conc_1 = concatenate([pool_1, conv_4])

        conv_5_1 = Conv2D(64, (1, 1), padding="same")(conc_1)
        conv_5_1 = BatchNormalization()(conv_5_1)
        conv_5_1 = Activation('relu')(conv_5_1)

        conv_6_1 = Conv2D(96, (3, 3), padding="valid")(conv_5_1)
        conv_6_1 = BatchNormalization()(conv_6_1)
        conv_6_1 = Activation('relu')(conv_6_1)

        conv_5_2 = Conv2D(64, (1, 1), padding="same")(conc_1)
        conv_5_2 = BatchNormalization()(conv_5_2)
        conv_5_2 = Activation('relu')(conv_5_2)

        conv_6_2 = Conv2D(64, (7, 1), padding="same")(conv_5_2)
        conv_6_2 = BatchNormalization()(conv_6_2)
        conv_6_2 = Activation('relu')(conv_6_2)

        conv_7_2 = Conv2D(64, (1, 7), padding="same")(conv_6_2)
        conv_7_2 = BatchNormalization()(conv_7_2)
        conv_7_2 = Activation('relu')(conv_7_2)

        conv_8_2 = Conv2D(96, (3, 3), padding="valid")(conv_7_2)
        conv_8_2 = BatchNormalization()(conv_8_2)
        conv_8_2 = Activation('relu')(conv_8_2)

        conc_2 = concatenate([conv_6_1, conv_8_2])

        conv_1_3 = Conv2D(192, (3,3),strides=2, padding="valid")(conc_2)
        conv_1_3 = BatchNormalization()(conv_1_3)
        conv_1_3 = Activation("relu")(conv_1_3)

        pool_1_4 = MaxPool2D(strides=2, padding = "valid")(conc_2)

        conc_3 = concatenate([conv_1_3, pool_1_4])

        return conc_3

    #fig 16
    def IRA(self, input_data):
        # relu_1 = Activation("relu")(input_data)
        b_0 = Conv2D(32, (1,1), padding="same")(input_data)
        b_0 = BatchNormalization()(b_0)
        b_0 = Activation("relu")(b_0)

        b_1 = Conv2D(32, (1, 1), padding="same")(input_data)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)
        b_1 = Conv2D(32, (3, 3), padding="same")(b_1)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)

        b_2 = Conv2D(32, (1, 1), padding="same")(input_data)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(48, (3, 3), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(64, (3, 3), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)

        conc_1 = concatenate([b_0, b_1, b_2])

        conv = Conv2D(384, (1,1), padding="same")(conc_1)

        out = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(conv)[1:],
               arguments={'scale': self.scale1})([input_data, conv])
        out = Activation("relu")(out)

        return out

    # fig 7
    # k l m n = 256 256 384 384

    def RA(self, input_data):
        b_0 = MaxPool2D((3,3), strides=2, padding="valid")(input_data)

        b_1 = Conv2D(384, (3,3), strides=2, padding="valid")(input_data)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)

        b_2 = Conv2D(256, (1,1), padding="same")(input_data)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(256, (3,3), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(384, (3,3), strides=2, padding="valid")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)

        conc_1 = concatenate([b_0, b_1, b_2])

        return conc_1

    # fig 17
    def IRB(self, input_data):
        b_1 = Conv2D(192, (1,1), padding="same")(input_data)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)

        b_2 = Conv2D(128, (1,1), padding="same")(input_data)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(160, (1,7), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(192, (7,1), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)

        conc_1 = concatenate([b_1, b_2])

        conv = Conv2D(1152, (1,1), padding="same")(conc_1)

        out = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(conv)[1:],
               arguments={'scale': self.scale2})([input_data, conv])
        out = Activation("relu")(out)
        return out


    def RB(self, input_data):
        b_0 = MaxPool2D((3,3), strides=2, padding="valid")(input_data)

        b_1 = Conv2D(256, (1,1), padding="same")(input_data)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)
        b_1 = Conv2D(384, (3,3),strides=2, padding="valid")(b_1)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)

        b_2 = Conv2D(256, (1,1), padding="same")(input_data)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(288, (3,3),strides=2, padding="valid")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)

        b_3 = Conv2D(256, (1,1), padding="same")(input_data)
        b_3 = BatchNormalization()(b_3)
        b_3 = Activation("relu")(b_3)
        b_3 = Conv2D(288, (3,3), padding="same")(b_3)
        b_3 = BatchNormalization()(b_3)
        b_3 = Activation("relu")(b_3)
        b_3 = Conv2D(320, (3,3),strides=2, padding="valid")(b_3)
        b_3 = BatchNormalization()(b_3)
        b_3 = Activation("relu")(b_3)

        conc = concatenate([b_0,b_1,b_2,b_3])
        return conc

    def IRC(self, input_data):
        b_1 = Conv2D(192, (1,1), padding="same")(input_data)
        b_1 = BatchNormalization()(b_1)
        b_1 = Activation("relu")(b_1)

        b_2 = Conv2D(192, (1,1), padding="same")(input_data)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(224, (1,3), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)
        b_2 = Conv2D(256, (3,1), padding="same")(b_2)
        b_2 = BatchNormalization()(b_2)
        b_2 = Activation("relu")(b_2)

        conc_1 = concatenate([b_1, b_2])

        conv = Conv2D(2144, (1,1), padding="same")(conc_1)
        out = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(conv)[1:],
               arguments={'scale': self.scale3})([input_data, conv])
        relu = Activation("relu")(out)
        return relu

class AlexNet():
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build_model(self):
        input_data = Input(self.input_shape)
        out = Conv2D(96,(11,11),strides=4,activation='relu')(input_data)
        out = MaxPool2D((3,3),strides=2)(out)
        out = BatchNormalization()(out)
        out = ZeroPadding2D(padding=2)(out)
        out = Conv2D(256, (5, 5), activation='relu',padding='same')(out)
        out = MaxPool2D((3,3), strides=2)(out)
        out = BatchNormalization()(out)
        out = ZeroPadding2D(padding=1)(out)
        out = Conv2D(384,(3,3),activation='relu',padding='same')(out)
        out = ZeroPadding2D(padding=1)(out)
        out = Conv2D(384, (3, 3), activation='relu',padding='same')(out)
        out = ZeroPadding2D(padding=1)(out)
        out = Conv2D(256, (3, 3), activation='relu',padding='same')(out)
        out = MaxPool2D((3,3), strides=2)(out)
        out = Flatten()(out)
        out = Dense(4096,activation="relu")(out)
        out = Dropout(rate=0.5)(out)
        out = Dense(4096,activation="relu")(out)
        out = Dropout(rate=0.5)(out)
        out = Dense(self.n_classes, activation="softmax")(out)
        model = Model(inputs=input_data, outputs = out)
        return model
    
class VGG16():
    # 224 224
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build_model(self):
        input_data = Input(self.input_shape)
        out = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(input_data)
        out = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(out)
        out = MaxPool2D(pool_size=(2,2),strides=(2,2))(out)
        out = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(out)
        out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
        out = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = MaxPool2D(pool_size=(2,2),strides=(2,2))(out)
        out = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = MaxPool2D(pool_size=(2,2),strides=(2,2))(out)
        out = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(out)
        out = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
        out = Flatten()(out)
        out = Dense(4096, activation="relu")(out)
        out = Dropout(rate=0.5)(out)
        out = Dense(4096, activation="relu")(out)
        out = Dropout(rate=0.5)(out)
        out = Dense(self.n_classes, activation="softmax")(out)
        model = Model(inputs=input_data, outputs=out)
        return model

    





















