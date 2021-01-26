
from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import Model

def classification_model(mattr, **kwargs):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    img_input = Input(shape=(mattr.inheight, mattr.inwidth, 3))
    x =SeparableConv2D(64, (3, 3), padding='same')(img_input)
    x =Activation('relu')(x)
    x =BatchNormalization()(x)
    x =Dropout(0.2)(x)

    x = SeparableConv2D(64, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SeparableConv2D(128, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(128, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(256, (3, 3), padding='same' )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(256 )(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(mattr.nclasses)(x)
    #x = Flatten()(x)
    
    #x = Reshape((mattr.outheight*mattr.outwidth, -1))(x)
    
    if(not mattr.from_logits):
        x = (Activation('softmax', name='softmax'))(x)

    model = Model( inputs=img_input , outputs=x )
    model.outwidth =1
    model.outheight = 1


    return model








    # #data augmentation
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    # # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)



    #optimization details
    # sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


    # training process in a for loop with learning rate drop every 25 epoches.

    return model



