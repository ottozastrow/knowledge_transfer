#! /usr/bin/env python3

import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import *
# from tf.keras import backend as K
from tensorflow.keras.layers import Layer
from metrics_and_losses import hardswish


def up_plain(num, x, channels, padding, connection=None):
    x = UpSampling2D((2, 2), name='up_block%i_upsample' % num)(x)
    x = SeparableConv2D(channels, (3, 3), padding=padding,
                        name="up_block%i_conv" % num)(x)
    x = BatchNormalization(axis=-1, name="up_block%i_batchnorm" % num)(x)
    return x


def down_unet(num, x, channels, padding, double_conv=False, activation="relu"):
    connection = x
    # activation function can be a locally defined fn or a str passed to keras
    activation_fn = hardswish if activation == "hardswish" else activation
    x = SeparableConv2D(
        channels, (3, 3), activation=activation_fn,
        padding=padding, name='down_block%i_conv1_' % num,
        kernel_initializer=glorot_normal())(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='down_block%i_pool' % num)(x)
    if(double_conv):
        x = SeparableConv2D(
            channels, (3, 3), activation=activation_fn,
            padding=padding, name='down_block%i_conv2_' % num,
            kernel_initializer=glorot_normal())(x)
    # x = Dropout(dropout_rate, name='down_block%i_dropout' % num)(x)
    x = BatchNormalization(axis=-1, name="down_block%i_batchnorm" % num)(x)
    return x, connection


def up_unet(num, x, channels, padding, connection):
    x = UpSampling2D((2, 2), name='up_block%i_upsample' % num)(x)
    x = Concatenate(axis=-1)([x, connection])
    x = SeparableConv2D(channels, (3, 3),
                        padding='same', name="up_block%i_conv" % num)(x)
    x = BatchNormalization(axis=-1, name="up_block%i_batchnorm" % num)(x)
    return x


def up_learnt(num, x, channels, padding, connection=None):
    # x = Conv2DTranspose(
    #             channels, (3,3), strides=2,
    #             padding=padding, kernel_initializer=glorot_normal(),
    #             use_bias=False)(x)
    x = Conv2DTranspose(channels//2, (1, 1), strides=2, padding=padding,
                        kernel_initializer=glorot_normal())(x)
    x = SeparableConv2D(channels, (2, 2), padding=padding)(x)

    x = BatchNormalization(axis=-1, name="up_block%i_batchnorm" % num)(x)
    return x


def down_segnet(num, x, channels, padding, dropout_rate=0.0):
    x = SeparableConv2D(channels, (3, 3), activation='relu',
                        padding=padding, name='down_block%i_conv2_' % num,
                        kernel_initializer=glorot_normal())(x)

    x, indices = MaxPoolingWithArgmax2D((2, 2))(x)
    # x = Dropout(dropout_rate, name='down_block%i_dropout' % num)(x)
    x = (BatchNormalization(axis=-1, name="down_block%i_batchnorm" % num))(x)
    return x, indices


def up_segnet(num, x, channels, padding, connection=None):
    x = MaxUnpooling2D((2, 2))([x, connection])
    x = SeparableConv2D(channels, (3, 3), padding=padding,
                        name="up_block%i_conv" % num)(x)
    x = BatchNormalization(axis=-1, name="up_block%i_batchnorm" % num)(x)
    return x


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=strides, padding=padding
        )

        argmax = tf.cast(argmax, tf.float32)
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]



class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.name_scope(self.name):
            mask = tf.cast(mask, 'int32')
            #input_shape = tf.shape(updates, out_type='int32')
            input_shape = updates.get_shape()

            # This statement is required if I don't want to specify a batch size
            if input_shape[0] == None:
                batches = 1
            else:
                batches = input_shape[0]

            #  calculation new shape
            if output_shape is None:
                output_shape = (
                        batches,
                        input_shape[1]*self.size[0],
                        input_shape[2]*self.size[1],
                        input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype='int32')
            batch_shape = tf.concat(
                    [[batches], [1], [1], [1]],
                    axis=0)
            batch_range = tf.reshape(
                    tf.range(output_shape[0], dtype='int32'),
                    shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(
                tf.stack([b, y, x, f]),
                [4, updates_size]))
            values = tf.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

def down_mobilenetv3(num, inputs, channels, padding, *, double_conv, dropout_rate=0.0):

    from Models.mobilenetv3.layers import Bneck
    exp_channels = channels * 4
    # from Models.mobilenetv3.utils import LayerNamespaceWrapper
    # x = (LayerNamespaceWrapper(
    #         Bneck(
    #             out_channels=channels,
    #             exp_channels=exp_channels,
    #             kernel_size=3,
    #             stride=2,
    #             use_se=True,
    #             act_layer='hswish',
    #         ),
    #         name=f"Bneck{num}")
    #     )(x)
    x = Bneck(
            out_channels=channels,
            exp_channels=exp_channels,
            kernel_size=3,
            stride=2,
            use_se=True,
            act_layer='hswish',
            num=num
        )(inputs)
    if(double_conv):
        x = Bneck(
            out_channels=channels,
            exp_channels=exp_channels,
            kernel_size=3,
            stride=1,
            use_se=True,
            act_layer='hswish',
            num=num + 0.1
        )(x)
    # pdb.set_trace()
    return x, inputs
