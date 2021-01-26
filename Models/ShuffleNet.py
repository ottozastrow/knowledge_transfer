from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
import numpy as np
import os
import json
file_path = os.path.dirname( os.path.abspath(__file__) )


def ShuffleNet(mattr, **kwargs):
	if('num_shuffle_units' not in kwargs.keys()):
		num_shuffle_units=[2,5]
	else:
		num_shuffle_units=kwargs['num_shuffle_units']

	if('decoder_size' not in kwargs.keys()):
		decoder_size=1
	else:
		decoder_size=kwargs['decoder_size']

	if('scale_factor' not in kwargs.keys()):
		scale_factor=0.5
	else:
		scale_factor=kwargs['scale_factor']
		
	if('bottleneck_ratio' not in kwargs.keys()):
		bottleneck_ratio=1.0
	else:
		bottleneck_ratio=kwargs['bottleneck_ratio'] 

	if('groups' not in kwargs.keys()):
		groups=2
	else:
		groups=kwargs['groups']



	def channel_split(x, name=''):
	    # equipartition
	    in_channles = x.shape.as_list()[-1]
	    ip = in_channles // groups
	    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
	    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
	    return c_hat, c

	def channel_shuffle(x):
	    height, width, channels = x.shape.as_list()[1:]
	    channels_per_split = channels // 2
	    x = K.reshape(x, [-1, height, width, groups, channels_per_split])
	    x = K.permute_dimensions(x, (0,1,2,4,3))
	    x = K.reshape(x, [-1, height, width, channels])
	    return x

	def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):

	    prefix = 'stage{}/block{}'.format(stage, block)
	    bottleneck_channels = int(out_channels * bottleneck_ratio)
	    if strides < 2:
	        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
	        inputs = c

	    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
	    x = BatchNormalization(axis=-1, name='{}/bn_1x1conv_1'.format(prefix))(x)
	    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
	    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
	    x = BatchNormalization(axis=-1, name='{}/bn_3x3dwconv'.format(prefix))(x)
	    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
	    x = BatchNormalization(axis=-1, name='{}/bn_1x1conv_2'.format(prefix))(x)
	    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

	    if strides < 2:
	        ret = Concatenate(axis=-1, name='{}/concat_1'.format(prefix))([x, c_hat])
	    else:
	        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
	        s2 = BatchNormalization(axis=-1, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
	        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
	        s2 = BatchNormalization(axis=-1, name='{}/bn_1x1conv_3'.format(prefix))(s2)
	        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
	        ret = Concatenate(axis=-1, name='{}/concat_2'.format(prefix))([x, s2])

	    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

	    return ret


	def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
	    x = shuffle_unit(x, out_channels=channel_map[stage-1],
	                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

	    for i in range(1, repeat+1):
	        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
	                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

	    return x

	out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}
	exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
	out_channels_in_stage = 2**exp
	out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
	out_channels_in_stage[0] = 24  # first stage has always 24 output channels
	out_channels_in_stage *= scale_factor
	out_channels_in_stage = out_channels_in_stage.astype(int)

	img_input = Input(shape=(mattr.inheight,mattr.inwidth,3))

	x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)

	x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)


	for stage in range(len(num_shuffle_units)):
		repeat = num_shuffle_units[stage]
		x = block(x, out_channels_in_stage,repeat=repeat,bottleneck_ratio=bottleneck_ratio,stage=stage + 2)

	# x = GlobalAveragePooling2D(name="global_pool")(x)

	# o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(x)
	o = ( SeparableConv2D(256, (3, 3), padding='valid', data_format='channels_last'))(x)
	o = ( BatchNormalization(axis=-1))(o)

	
	for r in range(decoder_size):

		o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
		# o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
		o = ( SeparableConv2D( 128, (3, 3), padding='same', data_format='channels_last'))(o)
		o = ( BatchNormalization(axis=-1))(o)

	o =  SeparableConv2D( mattr.nclasses , (3, 3) , padding='same', data_format='channels_last' )( o )

	o_shape = Model(img_input , o ).output_shape
	mattr.outheight = o_shape[1]
	mattr.outwidth = o_shape[2]

	# o = (Reshape((  -1, mattr.outheight*mattr.outwidth   )))(o)
	# o = (Permute((2, 1)))(o)

	o = (Reshape((  mattr.outheight*mattr.outwidth, mattr.nclasses   )))(o)
	o = (Activation('softmax', name='softmax'))(o)
	
	model = Model( img_input , o )
	model.outwidth = mattr.outwidth
	model.outheight = mattr.outheight

	return model

