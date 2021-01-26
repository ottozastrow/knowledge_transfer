from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
import tensorflow as tf

import os

''' alpha : The width multiplier for the model.
              The number of input and output channels at each layer are multiplied by this value.
            Default is 1.0, provides a quadratic reduction in complexity.

    depth_multiplier : The resolution multiplier. 
            The resolution of the input image and the subsequent internal represantations are multiplied by this value.
            Default is 1.0, provides a quadratic reduction in complexity.

    See the MobileNet paper for more info: https://arxiv.org/pdf/1704.04861.pdf
'''

def mobilenet(mattr,  kwargs={'alpha':1.0, 'depth_multiplier':1.0}):

    img_input = Input(shape=(mattr.inheight,mattr.inwidth,3))
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    alpha = kwargs["alpha"]
    depth_multiplier = kwargs["depth_multiplier"]
    
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _conv_block(inputs, filters, alpha=1, kernel=(3, 3), strides=(1, 1)):

        filters = int(filters * alpha)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
        x = Conv2D(filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        return ReLU(6., name='conv1_relu')(x)

    def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):

        in_channels = K.int_shape(inputs)[channel_axis]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = Conv2D(expansion * in_channels,kernel_size=1,padding='same',use_bias=False,activation=None,
                                               name=prefix + 'expand')(x)

            x = BatchNormalization(axis=channel_axis,epsilon=1e-3,momentum=0.999,name=prefix + 'expand_BN')(x)
            x = ReLU(6., name=prefix + 'expand_relu')(x)

        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if stride == 2:
            img_dim = 2 if K.image_data_format() == 'channels_first' else 1
            input_size = K.int_shape(x)[img_dim:(img_dim + 2)]

            if input_size[0] is None:
                adjust = (1,1)
            else:
                adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

            correct_padding = ((1 - adjust[0], 1), (1 - adjust[1], 1))

            x = ZeroPadding2D(padding=correct_padding, name=prefix + 'pad')(x)

        x = DepthwiseConv2D(kernel_size=3,strides=stride,activation=None, use_bias=False,
                                    padding='same' if stride == 1 else 'valid',name=prefix + 'depthwise')(x)

        x = BatchNormalization(axis=channel_axis,epsilon=1e-3,momentum=0.999,name=prefix + 'depthwise_BN')(x)

        x = ReLU(6., name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,kernel_size=1,padding='same',use_bias=False,activation=None,name=prefix + 'project')(x)
        x = BatchNormalization(axis=channel_axis,epsilon=1e-3,momentum=0.999,name=prefix + 'project_BN')(x)

        if in_channels == pointwise_filters and stride == 1:
            return Add(name=prefix + 'add')([inputs, x])
        return x


    x = _conv_block(img_input, 32, alpha, strides=(2, 2))

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    
    x = (ZeroPadding2D( (2,0) , data_format='channels_last'))(x)
    
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    #x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = (ZeroPadding2D( (2,0) , data_format='channels_last'))(x)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    #x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    #x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = (ZeroPadding2D( (2,0) , data_format='channels_last'))(x)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    #x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    #x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = (ZeroPadding2D( (2,0) , data_format='channels_last'))(x)

    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=6, block_id=14)
    #x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    
    #x = (ZeroPadding2D( (2,0) , data_format='channels_last'))(x)

    x = _inverted_res_block(x, filters=192, alpha=alpha, stride=1, expansion=6, block_id=16)

    # last_block_filters = 1280
    # x = Conv2D(last_block_filters,kernel_size=1,use_bias=False,name='Conv_1')(x)
    # x = BatchNormalization(axis=channel_axis,epsilon=1e-3,momentum=0.999,name='Conv_1_bn')(x)
    # x = ReLU(6., name='out_relu')(x)

    x = AveragePooling2D((7,7),strides =(1,1), name='avg_pool')(x)
    # x = Flatten(name='flatten')(x)
       # x = Dense( 1024, activation='relu', name='fully_connected')(x)
    # x = Dense( 1000, activation='softmax', name='predictions')(x)
    
    # DECODER 
    
    o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(x)
    o = ( SeparableConv2D(256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization(axis=-1))(o)

    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
    o = ( SeparableConv2D( 128, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization(axis=-1))(o)

    o =  SeparableConv2D( mattr.nclasses , (3, 3) , padding='same', data_format='channels_last' )( o )
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)

    o_shape = Model(img_input , o ).output_shape
    mattr.outheight = o_shape[1]
    mattr.outwidth = o_shape[2]

    # o = (Reshape((  -1, mattr.outheight*mattr.outwidth   )))(o)
    # o = (Permute((2, 1)))(o)

    o = (Reshape((  mattr.outheight*mattr.outwidth, -1   )))(o)
    o = (Activation('softmax', name='softmax'))(o)
    model = Model( img_input , o )
    model.outwidth = mattr.outwidth
    model.outheight = mattr.outheight

    return model

