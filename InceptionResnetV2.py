from keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout, concatenate
from keras.models import Model

class IRV2():
    def __init__(self, input_shape, n_classes, activation, p=1, t=2, r=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation

    def build_model(self):
        input_data = Input(shape=self.input_shape)
        stem = self.stem(input_data)
        flatten_1 = Flatten()(stem)
        final = Dense(self.n_classes, activation= self.activation)(flatten_1)
        model = Model(inputs = input_data, outputs = final)

        return model


    def stem(self, input_data):
        conv_1 = Conv2D(32, (3,3), strides=2, padding="valid")(input_data)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation('relu')(conv_1)

        conv_2 = Conv2D(32, (3, 3), padding="valid")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation('relu')(conv_2)

        conv_3 = Conv2D(32, (3, 3), padding="same")(conv_2)
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

        return conc_2


ir = IRV2()