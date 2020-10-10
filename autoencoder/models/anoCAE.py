"""
Model inspired by: 
https://github.com/ktjktj0911/MVTEC-anomaly-detection/blob/master/model2.ipynb
"""

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Dense,
    ZeroPadding2D,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Model

# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (227, 227)
# SHAPE = (483, 483)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN

# Training parameters
EARLY_STOPPING = 12
REDUCE_ON_PLATEAU = 6


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    inputs = Input(shape=img_dim)
    # inputs = Lambda(lambda x:x/255)(inputs)

    # -------------------Contraction-----------------------

    c1 = Conv2D(96, (11, 11), strides=4, activation="relu", padding="valid")(inputs)
    m1 = MaxPooling2D((3, 3), strides=2)(c1)

    c2 = ZeroPadding2D(padding=(2, 2))(m1)
    c2 = Conv2D(256, (5, 5), strides=1, activation="relu", padding="valid")(c2)
    m2 = MaxPooling2D((3, 3), strides=2)(c2)

    c3 = ZeroPadding2D(padding=(1, 1))(m2)
    c3 = Conv2D(384, (3, 3), strides=1, activation="relu", padding="valid")(c3)

    c4 = ZeroPadding2D(padding=(1, 1))(c3)
    c4 = Conv2D(384, (3, 3), strides=1, activation="relu", padding="valid")(c4)

    c5 = ZeroPadding2D(padding=(1, 1))(c4)
    c5 = Conv2D(256, (3, 3), strides=1, activation="relu", padding="valid")(c5)
    m5 = MaxPooling2D((3, 3), strides=2)(c5)

    # f = Flatten()(m5)
    d = Dense(2048, activation="relu")(m5)

    # ---------------------Expansion------------------------

    em6 = Conv2DTranspose(256, (3, 3), strides=2)(d)
    e6 = Conv2D(256, (3, 3), strides=1, activation="relu", padding="valid")(em6)
    e6 = ZeroPadding2D(padding=(1, 1))(e6)

    e7 = Conv2D(384, (3, 3), strides=1, activation="relu", padding="valid")(e6)
    e7 = ZeroPadding2D(padding=(1, 1))(e7)

    e8 = Conv2D(384, (3, 3), strides=1, activation="relu", padding="valid")(e7)
    e8 = ZeroPadding2D(padding=(1, 1))(e8)

    em9 = Conv2DTranspose(256, (3, 3), strides=2)(e8)
    e9 = Conv2D(256, (5, 5), strides=1, activation="relu", padding="valid")(em9)
    e9 = ZeroPadding2D(padding=(2, 2))(e9)

    em10 = Conv2DTranspose(96, (3, 3), strides=2)(e9)

    e10 = Conv2DTranspose(96, (11, 11), strides=4)(em10)
    e10 = Conv2D(96, (11, 11), strides=4, activation="relu", padding="valid")(e10)
    e10 = Conv2DTranspose(96, (11, 11), strides=4)(e10)
    outputs = Conv2D(channels, (1, 1), activation="sigmoid")(e10)

    autoencoder = Model(inputs=[inputs], outputs=[outputs])

    return autoencoder
