

import os
import time
from absl import app
from absl import flags
import tensorflow as tf # TF2
assert tf.__version__.startswith('2')


def model(mattr, **kwargs):
    """modified by ovz"""
    apply_dropout = False
    dropout_factor=0.0
    if("dropout" in kwargs.keys()):
      apply_dropout = True
      dropout_factor = kwargs["dropout"]
    alpha=1.0
    if("alpha" in kwargs.keys()):
      alpha = kwargs['alpha']

    model = tf.keras.applications.MobileNetV2(input_shape=[mattr.inheight, mattr.inwidth, 3], classes = mattr.nclasses, include_top=True, **kwargs)

    # lastlayer = model.layers[-1]
    # #x = tf.keras.layers.Reshape((mattr.outwidth*mattr.outheight, mattr.nclasses))(lastlayer)
    # x = tf.keras.layers.Activation('softmax', name='softmax')(lastlayer)
    # model = tf.keras.Model(inputs = model.inputs, outputs = x)

    return model
