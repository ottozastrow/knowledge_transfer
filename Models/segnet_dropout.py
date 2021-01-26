
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_normal
import os


import os 
# file_path = os.path.dirname( os.path.abspath(__file__) )


def SegmentationModel( mattr, **kwargs):
				# kwargs={'dropout_rate_a':0.0,'dropout_rate_b':0.0,'dropout_rate_c':0.0 }):

	# dropout_rate_a = kwargs['dropout_rate_a']
	# dropout_rate_b = kwargs['dropout_rate_b']
	# dropout_rate_c = kwargs['dropout_rate_c']
	dropout_rate_a = 0.0
	dropout_rate_b = 0.0
	dropout_rate_c = 0.0

	img_input = Input(shape=(mattr.inheight, mattr.inwidth, 3))

	# block 1
	x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(img_input)
	#x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last' )(x)

	# Block 2
	x = SeparableConv2D(64, (3, 3), strides= (2,2,), activation='relu', padding='same', name='block2_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last' )(x)
	x = Dropout(dropout_rate_a)(x)


	# Block 3
	x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last' )(x)
	x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last' )(x)
	x = Dropout(dropout_rate_b)(x)

	# upsample block 1
	x = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(x)
	x = ( SeparableConv2D(256, (3, 3), padding='valid', data_format='channels_last'))(x)
	x = Dropout(dropout_rate_c)(x)
	x = ( BatchNormalization(axis=-1))(x)

	#upsample block 2
	x = ( UpSampling2D( (2,2), data_format='channels_last'))(x)
	x = ( ZeroPadding2D( (1,1), data_format='channels_last'))(x)
	x = ( SeparableConv2D( 128, (3, 3), padding='valid', data_format='channels_last'))(x)
	x = ( BatchNormalization(axis=-1, name="batchnorm_attach_attention"))(x)

	x =  SeparableConv2D( mattr.nclasses , (3, 3) , padding='same', data_format='channels_last' )( x )
	o_shape = Model(img_input , x ).output_shape
	mattr.outheight = o_shape[1]
	mattr.outwidth = o_shape[2]

	# o = (Reshape((  -1, mattr.outheight*mattr.outwidth   )))(o)
	# o = (Permute((2, 1)))(o)

	x = (Reshape((  mattr.outheight*mattr.outwidth, mattr.nclasses  )))(x)

	
	if(not mattr.from_logits):
		x = (Activation('softmax', name='softmax'))(x)
	model = Model( inputs=img_input , outputs=x )
	model.outwidth = mattr.outwidth
	model.outheight = mattr.outheight

	model.inputHeight = 384
	model.inputWidth = 131

	return model

