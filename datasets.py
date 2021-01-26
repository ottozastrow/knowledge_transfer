import glob
import json
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import cv2
import preprocessing_tf


def convert_annotations2img(rootdir, mattr):
    for filename in tqdm.tqdm(
            list(glob.iglob(rootdir + '/annotations/**.tif.json',
                            recursive=True))):
        file = open(filename, "r")
        txt = file.read()
        img = json2mask(txt, mattr, filename)
        if img is not None:
            cv2.imwrite(filename[:-8] + "png", img)
        else:
            imgfilename = filename.replace("/annotations/", "/images/")[:-5]
            paths_exist = [os.path.exists(imgfilename),
                           os.path.exists(filename),
                           os.path.exists(filename[:-8] + "png")]
            if all(paths_exist):
                os.remove(filename)
                os.remove(filename[:-8] + "png")
                os.remove(imgfilename)
                print("deleted", filename, "and ", imgfilename)
            else:
                print(paths_exist, filename, imgfilename)
    for filename in tqdm.tqdm(list(glob.iglob(rootdir + "/images/**.jpg",
                                              recursive=True))):
        gt_filename = filename.replace("/images/", "/annotations/")[:-3]\
            + "tif.json"
        if not os.path.exists(gt_filename):
            print("will remove", filename)
            os.remove(filename)


def json2mask(txt, mattr, filepath):
    """ converts string to barcode mask image"""
    img = np.zeros((2048, 2448, 3),
                   dtype=np.uint8)
    info = json.loads(txt)['codes']
    for code in info:
        barcode_area = (slice(code['y0'], code['y1']),
                        slice(code['x0'], code['x1']), slice(0, 3))
        leny = barcode_area[0].stop - barcode_area[0].start
        lenx = barcode_area[1].stop - barcode_area[1].start
        img[barcode_area] = 1
        if leny * lenx > (2048 * 2448) / 16:  # if barcodearea larger than a
            # 16th of the original image
            return None
    return img


def load_barcodes(mattr, args):
    """ loads dataset in pascal voc format"""
    mattr.nclasses = 2
    annotations = "../tgb/segmentation_datasets/"\
        "finished_datasets/segment_dataset_v2/annotations/"
    images = "../tgb/segmentation_datasets/"\
        "finished_datasets/segment_dataset_v2/images/"
    samples = list(glob.iglob(annotations + "**.png"))
    split_index = int(len(samples) * args.traintest_split)
    ds = convert_files_to_tfdataset(annotations, images, mattr, args,
                                    on_value=128)
    train_dataset = ds.take(split_index)
    val_dataset = ds.skip(split_index)
    print("splitting into ", len(samples[:split_index]), " and ",
          len(samples[split_index:]), " for train and test set")

    return train_dataset, val_dataset, len(samples[:split_index]),\
        len(samples[split_index:]), mattr.nclasses


def load_barcodes_synth(mattr, args):
    mattr.nclasses = 2
    ds_path = "../tgb/yolo/localization/"
    # convert_annotations2img(ds_path, mattr)
    train_annotations_path = ds_path + "annotations_train/"
    train_images_path = ds_path + "images_train/"
    val_annotations_path = ds_path + "annotations_val/"
    val_images_path = ds_path + "images_val/"

    train_length = len(list(glob.iglob(train_annotations_path + "**.png")))
    val_length = len(list(glob.iglob(val_annotations_path + "**.png")))

    val_dataset = convert_files_to_tfdataset(
        val_annotations_path, val_images_path, mattr, args)

    train_dataset = convert_files_to_tfdataset(
        train_annotations_path, train_images_path, mattr, args)

    return train_dataset, val_dataset, train_length, val_length, mattr.nclasses


def load_lawnbot(mattr, args):
    mattr.nclasses = 2
    ds_path = "../image-analysis/dataset_current/dataset_own_v9.1/"
    val_annotations_path = ds_path + "annotations_val/"
    val_images_path = ds_path + "images_val/"

    train_annotations_path = ds_path + "annotations_train/"
    train_images_path = ds_path + "images_train/"

    TRAIN_LENGTH = len(os.listdir(train_annotations_path))
    VAL_LENGTH = len(os.listdir(val_annotations_path))

    val_dataset = convert_files_to_tfdataset(val_annotations_path,
                                             val_images_path, mattr, args)

    train_dataset = convert_files_to_tfdataset(train_annotations_path,
                                               train_images_path, mattr, args)

    # train_dataset = val_dataset #activate this line for debug purposes
    return train_dataset, val_dataset, TRAIN_LENGTH, VAL_LENGTH, mattr.nclasses


def load_cityscapes(mattr, args):

    mattr.nclasses = 30

    def process(datapoint):
        img = datapoint['image_left']
        gt = datapoint['segmentation_label']
        img = tf.image.resize(img, (mattr.inheight, mattr.inwidth))
        gt = tf.image.resize(gt, (mattr.outheight, mattr.outwidth))

        gt = tf.dtypes.cast(gt, tf.uint8)
        gt = tf.one_hot(gt, mattr.nclasses)
        gt = tf.reshape(gt, (mattr.outheight, mattr.outwidth, mattr.nclasses))

        return img, gt

    dataset, info = tfds.load(
        'cityscapes/semantic_segmentation',
        data_dir="~/image-analysis/dataset_current/cityscapes",
        with_info=True)
    TRAIN_LENGTH = info.splits['train'].num_examples
    VAL_LENGTH = info.splits['validation'].num_examples
    test_dataset = dataset['validation']
    train_dataset = dataset['train']
    if(args.debug):
        test_dataset = test_dataset.take(mattr.batch_size*2)
        train_dataset = train_dataset.take(mattr.batch_size*2)
    test_dataset = test_dataset.map(process)
    train_dataset = train_dataset.map(process)
    return (train_dataset, test_dataset,
            TRAIN_LENGTH, VAL_LENGTH, mattr.nclasses)


def convert_files_to_tfdataset(gtfilepath, imgfilepath, mattr, args,
                               on_value=1):
    # on_value is the pixel value of the barcode class in the mask png
    # find the on value of a two class dataset by using np.unique(mask_img)
    def process_path(file_path):

        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (mattr.inheight, mattr.inwidth))

        file_path_gt = tf.identity(file_path)
        file_path_gt = tf.strings.regex_replace((file_path_gt), ".jpg", ".png")
        file_path_gt = tf.strings.regex_replace((file_path_gt),
                                                "/images/", "/annotations/")
        file_path_gt = tf.strings.regex_replace((file_path_gt),
                                                "/images_", "/annotations_")

        gt = tf.io.read_file(file_path_gt)
        gt = tf.image.decode_jpeg(gt, channels=1)
        gt = tf.image.resize(gt, (mattr.outheight, mattr.outwidth))
        gt = tf.reshape(gt, (mattr.outheight, mattr.outwidth))

        gt = tf.dtypes.cast(gt, tf.uint8)
        gt = tf.one_hot(gt, mattr.nclasses)

        gt = tf.dtypes.cast(gt, tf.float32)
        gt = tf.reshape(gt, (mattr.outheight, mattr.outwidth, mattr.nclasses))

        return img, gt

    ds_unlabled = tf.data.Dataset.list_files(imgfilepath + "*.jpg")
    # activate for debug purposes
    # ds_unlabled = tf.data.Dataset.list_files(imgfil
    # epath + "picture2019-07-29 12:58:15.594564.jpg")

    if(args.debug):
        ds_unlabled = ds_unlabled.take(mattr.batch_size)
    ds_labled = ds_unlabled.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds_labled


def dataset_details(name):
    if(name == "lawnbot"):
        ds_path = "../image-analysis/dataset_current/dataset_own_v9.1/"
        val_annotations_path = ds_path + "annotations_val/"

        train_annotations_path = ds_path + "annotations_train/"

        TRAIN_LENGTH = len(os.listdir(train_annotations_path))
        VAL_LENGTH = len(os.listdir(val_annotations_path))
        return TRAIN_LENGTH, VAL_LENGTH

    elif(name == "cifar10"):
        return 50000, 10000
    elif(name == "cityscapes"):
        return 2975, 500


def load_cifar10(mattr, args):
    mattr.nclasses = 10

    def map_to_tuple(dict):
        label = dict['label']
        return dict['image'], label

    def map_one_hot(img, label):
        label = tf.one_hot(label, depth=mattr.nclasses)
        return img, label

    data, info = tfds.load("cifar10", with_info=True)
    train_data, test_data = data['train'], data['test']

    TRAIN_LENGTH = info.splits['train'].num_examples
    VAL_LENGTH = info.splits['test'].num_examples

    train_data = train_data.map(map_to_tuple)
    test_data = test_data.map(map_to_tuple)

    train_data = train_data.map(map_one_hot)
    test_data = test_data.map(map_one_hot)

    subsample_factor = 1

    train_data = train_data.take(TRAIN_LENGTH//subsample_factor)
    test_data = test_data.take(VAL_LENGTH//subsample_factor)

    VAL_LENGTH = VAL_LENGTH//subsample_factor
    TRAIN_LENGTH = TRAIN_LENGTH//subsample_factor

    return train_data, test_data, TRAIN_LENGTH, VAL_LENGTH, mattr.nclasses

def load_mnist(mattr, args):
    mattr.nclasses = 10

    def map_to_tuple(dict):
        label = dict['label']
        return dict['image'], label

    def map_one_hot(img, label):
        label = tf.one_hot(label, depth=mattr.nclasses)
        return img, label

    data, info = tfds.load("mnist", with_info=True)
    train_data, test_data = data['train'], data['test']

    TRAIN_LENGTH = info.splits['train'].num_examples
    VAL_LENGTH = info.splits['test'].num_examples

    train_data = train_data.map(map_to_tuple)
    test_data = test_data.map(map_to_tuple)

    train_data = train_data.map(map_one_hot)
    test_data = test_data.map(map_one_hot)

    return train_data, test_data, TRAIN_LENGTH, VAL_LENGTH, mattr.nclasses


def load_teacher(filepath, mattr, args, attention_map_size=(16, 32)):

    def _parse_image(example):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_tensor(example, out_type=tf.float32)

    def _parse_mask(example):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_tensor(example, out_type=tf.float32)

    def read_tfrecords(dirname):

        def unconcat_img(datapoint):
            image = tf.split(datapoint, axis=-1, num_or_size_splits=[
                3, mattr.nclasses//args.subclasses_per_class, 
                mattr.nclasses])[0]
            image = tf.reshape(image, (mattr.inheight, mattr.inwidth, 3))
            return image

        def unconcat_gt(datapoint):
            gt = tf.split(datapoint, axis=-1, num_or_size_splits=[3, mattr.nclasses//args.subclasses_per_class, mattr.nclasses])[1]
            gt = tf.reshape(gt, (mattr.inheight, mattr.inwidth, mattr.nclasses))
            gt = tf.image.resize(gt, (mattr.outheight, mattr.outwidth), tf.image.ResizeMethod.BILINEAR)
            gt = tf.reshape(gt, (mattr.outheight*mattr.outwidth, mattr.nclasses))
            return gt

        def unconcat_prediction(datapoint):
            prediction = tf.split(datapoint, axis=-1, num_or_size_splits=[3, mattr.nclasses//args.subclasses_per_class, mattr.nclasses])[2]
            prediction = tf.reshape(prediction, (mattr.inheight, mattr.inwidth, mattr.nclasses))
            prediction = tf.image.resize(prediction, (mattr.outheight, mattr.outwidth), tf.image.ResizeMethod.BILINEAR)
            prediction = tf.reshape(prediction, (mattr.outheight*mattr.outwidth, mattr.nclasses))
            return prediction

        def unconcat(datapoint):
            splits = tf.split(datapoint, axis=-1, num_or_size_splits=[3, mattr.nclasses//args.subclasses_per_class, mattr.nclasses])
            image = splits[0]
            gt = splits[1]
            prediction = splits[2]

            image = tf.reshape(image, (mattr.inheight, mattr.inwidth, 3))

            gt = tf.reshape(gt, (mattr.inheight, mattr.inwidth, mattr.nclasses//args.subclasses_per_class))
            gt = tf.image.resize(gt, (mattr.outheight, mattr.outwidth), tf.image.ResizeMethod.BILINEAR)
            gt = tf.reshape(gt, (mattr.outheight*mattr.outwidth, mattr.nclasses//args.subclasses_per_class))

            prediction = tf.reshape(prediction, (mattr.inheight, mattr.inwidth, mattr.nclasses))
            prediction = tf.image.resize(prediction, (mattr.outheight, mattr.outwidth), tf.image.ResizeMethod.BILINEAR)
            prediction = tf.reshape(prediction, (mattr.outheight*mattr.outwidth, mattr.nclasses))

            return image, (gt, prediction)

        def unconcat_AT(datapoint):
            num_or_size_splits = [3, 1, mattr.nclasses//args.subclasses_per_class, mattr.nclasses]
            splits = tf.split(datapoint, axis=-1, num_or_size_splits=num_or_size_splits)
            image = splits[0]
            attention_map = splits[1]
            gt = splits[2]
            prediction = splits[3]

            image = tf.reshape(image, (mattr.inheight, mattr.inwidth, 3))

            gt = tf.reshape(gt, (mattr.inheight, mattr.inwidth, mattr.nclasses//args.subclasses_per_class))
            gt = tf.image.resize(gt, (mattr.outheight, mattr.outwidth), tf.image.ResizeMethod.BILINEAR)
            gt = tf.reshape(gt, (mattr.outheight*mattr.outwidth, mattr.nclasses//args.subclasses_per_class))

            prediction = tf.reshape(prediction, (mattr.inheight, mattr.inwidth, mattr.nclasses))
            prediction = tf.image.resize(prediction, (mattr.outheight, mattr.outwidth), tf.image.ResizeMethod.BILINEAR)
            prediction = tf.reshape(prediction, (mattr.outheight*mattr.outwidth, mattr.nclasses))

            attention_map = tf.reshape(attention_map, (mattr.inheight, mattr.inwidth, 1))
            attention_map = tf.image.resize(attention_map, (mattr.outheight, mattr.outwidth))
            attention_map = tf.reshape(attention_map, (mattr.outheight, mattr.outwidth))

            attention_map = tf.reshape(attention_map, (mattr.inheight, mattr.inwidth, 1))
            attention_map = tf.image.resize(attention_map, attention_map_size)
            attention_map = tf.reshape(attention_map, attention_map_size)

            return image, (gt, prediction, attention_map)

        files = tf.data.Dataset.list_files(dirname + "*.tfrecord")
        if(args.debug):  # enable fast debugging
            files = files.take(1)
        ds = tf.data.TFRecordDataset(files)

        ds = ds.map(_parse_mask)
        if(args.AT):
            ds = ds.map(unconcat_AT)
        else:
            ds = ds.map(unconcat)

        return ds

    test_dataset = read_tfrecords(filepath + "test/")
    train_dataset = read_tfrecords(filepath + "train/")

    return train_dataset, test_dataset


def prepare_dataset(train_dataset, test_dataset, TRAIN_LENGTH,
                    mattr, args, records_from_teacher):

    def normalize(input_image, label):
        if(args.image_normalization=="sub_mean"):
            input_image = preprocessing_tf.normalize_sub_mean(input_image, args.dataset)
        elif(args.image_normalization=="per_image_standardization"):
            input_image = preprocessing_tf.normalize_tfimage(input_image, args.dataset)
        elif(args.image_normalization=="divide"):
            input_image = preprocessing_tf.normalize_divide(input_image, args.dataset)
        input_image = tf.image.resize(input_image, (mattr.inheight, mattr.inwidth))

        return input_image, label

    def augment_segmentation(input_image, input_mask):
        # input_image, input_mask = preprocessing_tf.flip_horizontal(
        #   input_image, input_mask, mattr)

        zoom_range_x = args.zoom_intensity
        zoom_range_y = args.zoom_intensity

        # input_image, input_mask = preprocessing_tf.zoom(
        #     input_image, input_mask, zoom_range_x, zoom_range_y, mattr)
        return input_image, input_mask

    def augment_classifier(input_image, label):
        zoom_range_x = args.zoom_intensity
        zoom_range_y = args.zoom_intensity

        if(args.augment):
            input_image = preprocessing_tf.flip_classifier(input_image)
            input_image = preprocessing_tf.zoom_classifier(
                input_image, zoom_range_x, zoom_range_y, mattr)

        input_image = tf.reshape(input_image,
                                 (mattr.inheight, mattr.inwidth, 3))

        return input_image, label

    def flatten_segmenation_masks(input_image, input_mask):
        if(records_from_teacher):
            mask_hard = tf.reshape(
                input_mask[0], (mattr.outheight * mattr.outwidth,
                                mattr.nclasses//args.subclasses_per_class))
            mask_soft = tf.reshape(
                input_mask[1],
                (mattr.outheight * mattr.outwidth, mattr.nclasses))
            if(not args.AT):
                input_mask = (mask_hard, mask_soft)
            if(args.AT):
                attention_map = input_mask[2]
                input_mask = (mask_hard, mask_soft, attention_map)
        else:
            input_mask = tf.reshape(
                input_mask, (mattr.outheight * mattr.outwidth, mattr.nclasses))

        return input_image, input_mask

    num_parallel_calls = tf.data.experimental.AUTOTUNE

    if(args.debug):  # make dataset small for faster debugging
        train_dataset = train_dataset.take(8)
        test_dataset = test_dataset.take(8)

    # activate for debug purposes
    # minidb = test_dataset.take(1)
    # for image, label in minidb:
    #     visualization_tools.debugvisualize(image, label, args, mattr)
    # minidb = minidb.map(augment_classifier, num_parallel_calls=num_parallel_calls)
    # for image, label in minidb:
    #     visualization_tools.debugvisualize(image, label, args, mattr)

    test_dataset = test_dataset.map(normalize,
                                    num_parallel_calls=num_parallel_calls)
    train_dataset = train_dataset.map(normalize,
                                      num_parallel_calls=num_parallel_calls)
    if(not args.classification):
        test_dataset = test_dataset.map(
            flatten_segmenation_masks, num_parallel_calls=num_parallel_calls)
        train_dataset = train_dataset.map(
            flatten_segmenation_masks, num_parallel_calls=num_parallel_calls)
        preparing_teacher = args.distillation == "hinton_distillation"\
            and not records_from_teacher
        if(args.augment and not preparing_teacher):
            train_dataset = train_dataset.map(
                augment_segmentation, num_parallel_calls=num_parallel_calls)
            train_dataset = train_dataset.batch(mattr.batch_size,
                                                drop_remainder=True)
            list_images, list_labels = [], []
            for images, labels in train_dataset:
                images, labels = preprocessing_tf.cutmix(
                    mattr, images, labels, PROBABILITY=0.5,
                    records_from_teacher=records_from_teacher)
                list_images.append((images))
                list_labels.append((labels))
            images_tensor = tf.convert_to_tensor(list_images)
            if not records_from_teacher:
                labels_tensor = tf.convert_to_tensor(list_labels)
                tensor_tuples = (images_tensor, labels_tensor)
            else:
                list_gt, list_teacher = tuple(map(list, zip(*list_labels)))
                labels_tensor = tf.convert_to_tensor(list_gt)
                teacher_tensor = tf.convert_to_tensor(list_teacher)
                tensor_tuples = (images_tensor,
                                 (labels_tensor, teacher_tensor))
            train_dataset = tf.data.Dataset.from_tensor_slices(tensor_tuples)
            train_dataset = train_dataset.unbatch()

    else:
        if(args.augment):
            train_dataset = train_dataset.map(
                augment_classifier, num_parallel_calls=num_parallel_calls)
    test_dataset = test_dataset.cache().repeat().batch(mattr.batch_size,
                                                       drop_remainder=True)
    test_dataset = test_dataset.prefetch(buffer_size=num_parallel_calls)

    train_dataset = train_dataset.cache().repeat()
    if(not args.debug):
        train_dataset = train_dataset.shuffle(
            TRAIN_LENGTH, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(mattr.batch_size, drop_remainder=True)\
        .prefetch(buffer_size=num_parallel_calls)

    # for dataset debugging
    if False:
        import matplotlib
        matplotlib.use("TkAgg")
        for t in train_dataset.take(1):
            img = tf.reshape(t[1][0],
                             (mattr.outheight, mattr.outwidth, mattr.nclasses))

            plt.imshow(img[:, :, 0])
            plt.show()
            break
    return train_dataset, test_dataset
