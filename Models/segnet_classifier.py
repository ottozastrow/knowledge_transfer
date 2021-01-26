
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_normal
import os

import os 
# file_path = os.path.dirname( os.path.abspath(__file__) )


def classification_model( mattr, **kwargs):
	# kwargs={'dropout_rate_a':0.0,'dropout_rate_b':0.0,'dropout_rate_c':0.0 }):

	# dropout_rate_a = kwargs['dropout_rate_a']
	# dropout_rate_b = kwargs['dropout_rate_b']
	# dropout_rate_c = kwargs['dropout_rate_c']
	dropout_rate_a = 0.0
	dropout_rate_b = 0.0
	dropout_rate_c = 0.0

	img_input = Input(shape=(mattr.inheight, mattr.inwidth, 3))

	x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(img_input)
	x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(img_input)
	#x = SeparableConv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last' )(x)

	# Block 2
	x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = Dropout(dropout_rate_a)(x)
	x = ( BatchNormalization(axis=-1))(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last' )(x)


	# Block 3
	x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	# x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last' )(x)
	# x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last' )(x)
	x = Dropout(dropout_rate_b)(x)
	x = ( BatchNormalization(axis=-1))(x)

	# Block 4
	x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	# x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = Dropout(dropout_rate_a)(x)
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last' )(x)

	# Block 5
	x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	# x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = Dropout(dropout_rate_a)(x)
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last' )(x)

	# Block 6
	x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	# x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2', data_format='channels_last' , kernel_initializer=glorot_normal())(x)
	x = Dropout(dropout_rate_a)(x)
	x = BatchNormalization(axis=-1)(x)
	# x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool', data_format='channels_last' )(x)


	# x = ( SeparableConv2D(256, (3, 3), padding='valid', data_format='channels_last'))(x)
	# x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last' )(x)

	# x = Dropout(dropout_rate_c)(x)

	# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last' )(x)


	x = Flatten(name='flatten')(x)
	x = Dense(512, activation='relu', name='fc1')(x)
	x = Dense(mattr.nclasses, activation='relu', name='fc2')(x)


	# o = (Reshape((  -1, mattr.outheight*mattr.outwidth   )))(o)
	# o = (Permute((2, 1)))(o)

	# x = (Reshape((  mattr.outheight*mattr.outwidth, mattr.nclasses  )))(x)

	
	if(not mattr.from_logits):
		x = (Activation('softmax', name='softmax'))(x)
	model = Model( inputs=img_input , outputs=x )
	model.outwidth =1
	model.outheight = 1


	return model

