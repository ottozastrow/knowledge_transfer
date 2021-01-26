# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pix2pix.
"""


import os
import time
from absl import app
from absl import flags
import tensorflow as tf # TF2
assert tf.__version__.startswith('2')

# FLAGS = flags.FLAGS

# flags.DEFINE_integer('buffer_size', 400, 'Shuffle buffer size')
# flags.DEFINE_integer('batch_size', 1, 'Batch Size')
# flags.DEFINE_integer('epochs', 1, 'Number of epochs')
# flags.DEFINE_string('path', None, 'Path to the data folder')
# flags.DEFINE_boolean('enable_function', True, 'Enable Function?')


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.

  Conv2D => Batchnorm => LeakyRelu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer

  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      # result.add(InstanceNormalization())
      pass

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False, dropout_factor=0.5):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(dropout_factor))

  result.add(tf.keras.layers.ReLU())

  return result




def unet_model(mattr, **kwargs):

    """modified by ovz"""
    apply_dropout = False
    dropout_factor=0.0
    if("dropout" in kwargs.keys()):
      apply_dropout = True
      dropout_factor = kwargs["dropout"]
    alpha=1.0
    if("alpha" in kwargs.keys()):
      alpha = kwargs['alpha']

    base_model = tf.keras.applications.MobileNetV2(input_shape=[mattr.inheight, mattr.inwidth, 3], include_top=False, **kwargs)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    up_stack = [
        upsample(int(256 * alpha), 3, apply_dropout=True, dropout_factor=dropout_factor),  # 4x4 -> 8x8
        upsample(int(128 * alpha), 3, apply_dropout=True, dropout_factor=dropout_factor),  # 8x8 -> 16x16
        upsample(int(64 * alpha), 3, apply_dropout=True, dropout_factor=dropout_factor),  # 16x16 -> 32x32
        upsample(int(32 * alpha), 3, apply_dropout=True, dropout_factor=dropout_factor),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[mattr.inheight, mattr.inwidth, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])


    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        mattr.nclasses, 3, strides=2,
        padding='same', activation=None)  #64x64 -> 128x128


    x = last(x)

    x = tf.keras.layers.Reshape((mattr.outwidth*mattr.outheight, mattr.nclasses))(x)

    x = tf.keras.layers.Activation('softmax', name='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
