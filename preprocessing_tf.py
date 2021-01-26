import tensorflow as tf
import matplotlib.pyplot as plt


def zoom(input_image, input_mask, yrange, xrange, mattr):
    yzoom = (tf.random.uniform([1])*yrange)[0]
    xzoom = (tf.random.uniform([1])*xrange)[0]

    new_height = int(1/(1+yzoom) * mattr.inheight)
    new_width = int(1/(1+xzoom) * mattr.inwidth)

    # because the same random
    # transformation needs to be applied to both image and mask,
    # they are concatenated before the transform
    input_mask = tf.reshape(input_mask,
                            (mattr.outheight, mattr.outwidth, mattr.nclasses))
    input_mask = tf.image.resize(input_mask,
                                 (mattr.inheight, mattr.inwidth),
                                 method="nearest")
    concat = tf.concat([input_image, input_mask], axis=-1)

    concat = tf.image.random_crop(concat,
                                  [new_height, new_width, 3 + mattr.nclasses])
    input_image = tf.image.resize(concat[:, :,:3],
                                  (mattr.inheight, mattr.inwidth))
    input_mask = tf.image.resize(concat[:, :, 3:],
                                 (mattr.outheight, mattr.outwidth),
                                 method="nearest")
    input_mask = tf.reshape(input_mask,
                            (mattr.outheight *mattr.outwidth, mattr.nclasses))

    return input_image, input_mask


def zoom_classifier(input_image, yrange, xrange, mattr):
    yzoom = (tf.random.uniform([1])*yrange)[0]
    xzoom = (tf.random.uniform([1])*xrange)[0]
    new_height = int(1/(1+yzoom) * mattr.inheight)
    new_width = int(1/(1+xzoom) * mattr.inwidth)

    input_image = tf.image.random_crop(input_image, [new_height, new_width, 3])
    input_image = tf.image.resize(input_image, (mattr.inheight, mattr.inwidth))

    return input_image


def flip_horizontal(input_image, input_mask, mattr):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.reshape(input_mask,
                                (mattr.outheight, mattr.outwidth, mattr.nclasses))
        input_mask = tf.image.flip_left_right(input_mask)
        input_mask = tf.reshape(input_mask,
                                (mattr.outheight * mattr.outwidth, mattr.nclasses))

    return input_image, input_mask


def flip_classifier(input_image):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
    # if(tf.random.uniform(()) > 0.5):
    #     input_image = tf.image.flip_up_down(input_image)
    return input_image


def normalize_tfimage(input_image, dataset_name):
    return tf.image.per_image_standardization(input_image)


def normalize_sub_mean(input_image, dataset_name):
    # mean r: 102.43793473080285
    # mean_g: 170.1677151882308
    # mean_b: 167.4679635012596

    if(dataset_name == "lawnbot"):
        mean = tf.constant([102.4, 170.2, 176.5], dtype=tf.float32)
    elif(dataset_name == "cifar10"):
        mean = tf.constant([125.3, 123.0, 113.9], dtype=tf.float32)
        # mean = tf.constant([125.3, 113.9, 123.0], dtype=tf.float32)
    elif(dataset_name == "cityscapes"):
        mean = tf.constant([72.78044, 83.21195, 73.45286], dtype=tf.float32)
    elif(dataset_name == "barcodes"):
        mean = tf.constant([30, 30, 30], dtype=tf.float32)

    mean = tf.reshape(mean, [1, 1, 3])
    return (tf.cast(input_image, tf.float32) - mean) / 255.0


def denormalize_sub_mean(input_image, dataset_name):
    if(dataset_name == "lawnbot"):
        mean = tf.constant([102.4, 170.2, 176.5], dtype=tf.float32)
    elif(dataset_name == "cifar10"):
        mean = tf.constant([125.3, 123.0, 113.9], dtype=tf.float32)
    elif(dataset_name == "cityscapes"):
        mean = tf.constant([72.78044, 83.21195, 73.45286], dtype=tf.float32)
    elif(dataset_name == "barcodes"):
        mean = tf.constant([30, 30, 30], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])

    return tf.cast(input_image*255.0 + mean, tf.uint8)


def normalize_divide(input_image, dataset_name):
    return tf.cast(input_image, tf.float32) / 255.0

# def denormalize(input_image):
#     return tf.cast(input_image*255.0, tf.uint8)


def cutmix(mattr, image, label, PROBABILITY=1.0, records_from_teacher=False):
    # input image - is a batch of images of
    # size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    AUG_BATCH = mattr.batch_size
    CLASSES = mattr.nclasses
    DIM = mattr.inheight, mattr.inwidth
    if records_from_teacher:
        label = tf.concat(list(label), axis=-1)  # concat teacher and gt mask
        CLASSES *= 2
    label = tf.reshape(label,
                       (AUG_BATCH, mattr.outheight, mattr.outwidth, CLASSES))
    label = tf.image.resize(label, (DIM),
                            tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # matplotlib.backend("agg")
    imgs = []
    labs = []
    for j in range(AUG_BATCH):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, mattr.inwidth), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, mattr.inheight), tf.int32)
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(mattr.inwidth * tf.math.sqrt(1-b), tf.int32) * P
        HEIGHT = tf.cast(mattr.inheight * tf.math.sqrt(1-b), tf.int32) * P

        ya = tf.math.maximum(0,y-HEIGHT//2)
        yb = tf.math.minimum(mattr.inheight,y+HEIGHT//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(mattr.inwidth,x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:mattr.inwidth,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle,
                         image[j, yb:mattr.inheight, :, :]], axis=0)
        imgs.append(img)
        l_one = label[j,ya:yb,0:xa,:]
        l_two = label[k,ya:yb,xa:xb,:]
        l_three = label[j,ya:yb,xb:mattr.inwidth,:]
        l_middle = tf.concat([l_one, l_two, l_three], axis=1)
        lab = tf.concat([label[j, 0:ya, :, :],
                         l_middle, label[j, yb:mattr.inheight, :, :]], axis=0)
        labs.append(lab)
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE
    # OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),
                        (AUG_BATCH, mattr.inheight, mattr.inwidth, 3))
    label2 = tf.reshape(tf.stack(labs),
                        (AUG_BATCH, mattr.inheight, mattr.inwidth, CLASSES))
    label2 = tf.image.resize(label2,
                             (mattr.outheight, mattr.outwidth),
                             tf.image.ResizeMethod.BILINEAR)
    label2 = tf.reshape(label2, (AUG_BATCH, mattr.outheight * mattr.outwidth,
                                 CLASSES))
    if records_from_teacher:
        label2 = label2[:, :, CLASSES//2:], label2[:, :, :CLASSES//2]

    return image2, label2
