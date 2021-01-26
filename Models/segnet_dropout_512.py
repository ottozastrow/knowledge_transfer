
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_normal
import os

# file_path = os.path.dirname( os.path.abspath(__file__) )


def SegmentationModel( mattr, **kwargs):
	# kwargs={'dropout_rate_a':0.0,'dropout_rate_b':0.0,'dropout_rate_c':0.0 }):

	# dropout_rate_a = kwargs['dropout_rate_a']
	# dropout_rate_b = kwargs['dropout_rate_b']
	# dropout_rate_c = kwargs['dropout_rate_c']
	dropout_rate_a = 0.2
	dropout_rate_b = 0.2
	dropout_rate_c = 0.2

	img_input = Input(shape=(mattr.inheight,mattr.inwidth,3))

	x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(img_input)
	#x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last' )(x)

	# Block 2
	x = SeparableConv2D(128, (3, 3), strides= (2,2,), activation='relu', padding='same', name='block2_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last' )(x)
	x = Dropout(dropout_rate_a)(x)

	x = SeparableConv2D(256, (3, 3), strides= (2,2,), activation='relu', padding='same', name='block2_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last' )(x)
	x = Dropout(dropout_rate_a)(x)


	x = ( UpSampling2D( (2,2), data_format='channels_last'))(x)
	x = ( SeparableConv2D( 256, (3, 3), padding='same', data_format='channels_last'))(x)
	x = ( BatchNormalization(axis=-1))(x)


	# x = ( UpSampling2D( (2,2), data_format='channels_last'))(x)
	# x = ( SeparableConv2D( 256, (3, 3), padding='same', data_format='channels_last'))(x)
	# x = ( BatchNormalization(axis=-1))(x)

	x = ( UpSampling2D( (2,2), data_format='channels_last'))(x)
	x = ( SeparableConv2D( 128, (3, 3), padding='same', data_format='channels_last'))(x)
	x = ( BatchNormalization(axis=-1))(x)

	x = ( UpSampling2D( (2,2), name='up2d', data_format='channels_last'))(x)
	x = ( SeparableConv2D( 64, (3, 3), padding='same', data_format='channels_last'))(x)
	x = ( BatchNormalization(axis=-1))(x)

	x = ( UpSampling2D( (2,2), name='up2d_2', data_format='channels_last'))(x)
	x = ( SeparableConv2D( 32, (3, 3), padding='same', data_format='channels_last'))(x)
	x = ( BatchNormalization(axis=-1))(x)

	x = ( UpSampling2D( (2,2), name='up2d_3', data_format='channels_last'))(x)
	x = ( SeparableConv2D( 16, (3, 3), padding='same', data_format='channels_last'))(x)


	x =  SeparableConv2D( mattr.nclasses , (3, 3) , padding='same', data_format='channels_last' )( x )
	o_shape = Model(img_input , x ).output_shape
	mattr.outheight = o_shape[1]
	mattr.outwidth = o_shape[2]

	# o = (Reshape((  -1, mattr.outheight*mattr.outwidth   )))(o)
	# o = (Permute((2, 1)))(o)

	x = (Reshape((  mattr.outheight*mattr.outwidth, -1   )))(x)

	if(not mattr.from_logits):
		x = (Activation('softmax', name='softmax'))(x)
	model = Model( img_input , x )
	model.outwidth = mattr.outwidth
	model.outheight = mattr.outheight

	return model

