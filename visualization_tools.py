import tensorflow as tf
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import wandb
import os
import wandb
from datetime import datetime
import numpy as np
import cv2
import preprocessing_tf
import tensorflow.keras.backend as K
from time import time


def measure_latency(model, test_dataset,
                    disable_tflite_inference=True, batch_size=1):
    if(not disable_tflite_inference):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    num_samples = 30
    samples = test_dataset.take(num_samples//batch_size).unbatch().batch(1)
    times = []
    if not disable_tflite_inference:
        samples = [sample[0].numpy() for sample in samples]
    # tf.profiler.experimental.start('logs/inference_logs')
    for sample in samples:
        tic = time()
        if(not disable_tflite_inference):
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
        else:
            model.predict(sample)
        toc = time()
        times.append((toc-tic))

        # sample = preprocessing_tf.denormalize_sub_mean(sample[0], "lawnbot")
        # visualization_tools.debugvisualize(sample, output_data, args, mattr)
        # tf.profiler.experimental.stop()
    times = times[5:]  # remove warmup phase
    times = np.array(times)
    avg = times.mean()
    return avg


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, mattr, args, test_dataset):
        self.mattr = mattr
        self.args = args
        self.test_dataset = test_dataset.take(4)

    def on_epoch_begin(self, epoch, logs=None):
        if((epoch) % 1 == 0):
            if(self.args.classification):
                visualize_classifier(self.mattr, self.test_dataset,
                                     args=self.args, num=1,
                                     epoch=epoch, model=self.model)
            else:
                show_predictions(mattr=self.mattr, dataset=self.test_dataset,
                                 args=self.args, epoch=epoch,
                                 model=self.model)
                print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


def visualize_classifier(mattr, dataset, args, epoch=-1, num=1, model=None):
    outpath = "visualizations/" + str(datetime.now()) + "/"
    counter = 0
    orig_classes = mattr.nclasses//args.subclasses_per_class
    buckets = [[] for _ in range(orig_classes)]
    buckets_subclasses = [[] for _ in range(mattr.nclasses)]
    gt_buckets = [[] for _ in range(orig_classes)]

    for batch in dataset.take(num):  # num of batches
        img_batch = batch[0]
        gt_batch = batch[1]
        logits_batch = model.predict(img_batch)

        image_orig_batch = preprocessing_tf.denormalize_sub_mean(
            img_batch, args.dataset)

        logits_origclass_batch = tf.reshape(
            logits_batch,
            (mattr.batch_size, orig_classes, args.subclasses_per_class))
        logits_origclass_batch = K.sum(logits_origclass_batch, axis=-1)
        predictions_subclass_batch = K.argmax(K.softmax(logits_batch), axis=-1)
        predictions_batch = K.argmax(K.softmax(logits_origclass_batch),
                                     axis=-1)
        gt_batch = K.argmax(gt_batch, axis=-1)

        for c in range(orig_classes):
            current_origs = [i for i in range(mattr.batch_size)
                             if predictions_batch[i] == c]
            buckets[c] = current_origs

            gt_current_origs = [i for i in range(mattr.batch_size)
                                if gt_batch[i] == c]
            gt_buckets[c] = gt_current_origs

        for c in range(mattr.nclasses):
            current_origs = [i for i in range(mattr.batch_size)
                             if predictions_subclass_batch[i] == c]
            buckets_subclasses[c] = current_origs

        matches = [(predictions_batch[i] == gt_batch[i])
                   for i in range(len(predictions_batch))]

        visualize_buckets(image_orig_batch,
                          buckets, matches, buckets_subclasses)


def visualize_buckets(images, buckets, matches, buckets_subclasses):
    buckets = buckets_subclasses

    num_buckets_to_visualize = min(10, len(buckets))
    num_subclass_buckets_to_visualize = min(10, len(buckets))
    buckets = buckets[10:10+num_buckets_to_visualize]
    buckets_subclasses = buckets_subclasses[:num_subclass_buckets_to_visualize]


    bucket_lengths = [len(b) for b in buckets]
    longest_bucket = min(10, max(bucket_lengths))
    redflag = (np.ones((32, 4, 3)) * np.array([255, 0, 0])).astype(np.uint8)
    greenflag = (np.ones((32, 4, 3)) * np.array([0, 255, 0])).astype(np.uint8)

    fig = plt.figure()
    print(buckets)
    for i in range(num_buckets_to_visualize):
      counter = 1
      for j in range(len(buckets[i])):
        if(j>=longest_bucket):
          break
        plt.subplot(num_buckets_to_visualize, longest_bucket, i*longest_bucket + counter)
        index = buckets[i][j]

        img = images[index].numpy()
        match = matches[index]
        if(not match):
          img = np.concatenate([img, redflag], axis=1)
          #img[:,:,0] = np.pad(img[:,:,0], 2, 'constant', constant_values=(0,))
        else:
          #img[:,:,0] = np.pad(img[:,:,0], 2, 'constant', constant_values=(255,))
          img = np.concatenate([img, greenflag], axis=1)

        plt.imshow(img)
        plt.axis('off')
        counter+=1
    plt.show()


def show_predictions(mattr, dataset, args, epoch=-1, model=None):
    upload_wandb = (not args.disable_wandb) and (epoch != -1)
    includes_soft_mask = False
    outpath = "visualizations/" + args.experiment_name\
        + "/" + str(datetime.now()) + "/"
    counter = 0
    for sample in dataset.take(args.num_visualize):
        counter += 1
        image = sample[0][0]
        gt = sample[1][0]
        # if distillation is used the data has shape
        # (image, (gt, gt_soft)) instead of (image, gt)
        if(tf.rank(gt) == 3):
            includes_soft_mask = True
            gt = sample[1][0][0]
            gt_soft = sample[1][1][0]
            gt_soft = tf.reshape(
                gt_soft, (mattr.outheight, mattr.outwidth, mattr.nclasses))

            # gt_soft = tf.argmax(gt_soft, axis=-1)
            gt_soft = gt_soft[:, :, 0]
            # gt_soft = tf.reshape(gt_soft,
            #  (mattr.outheight, mattr.outwidth, mattr.nclasses))[:,:,0]
        print("includes soft is ", includes_soft_mask)
        image_orig = preprocessing_tf.denormalize_sub_mean(image, args.dataset)
        # predict wants a 4D input Tensor, ()
        image = tf.reshape(image, (1, mattr.inheight, mattr.inwidth, 3))

        gt = tf.reshape(gt, (mattr.outheight, mattr.outwidth,
                             mattr.nclasses//args.subclasses_per_class))
        # gt = gt[:,:,0]
        gt = tf.argmax(gt, axis=-1)
        model_output = model.predict(image)

        pred_mask = tf.reshape(
            model_output[0], (mattr.outheight, mattr.outwidth, mattr.nclasses))

        pred_mask = tf.argmax(pred_mask, axis=-1)

        print("plotting figure")

        display_list = [image_orig, gt, pred_mask]
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        if(includes_soft_mask):
            title.append("Soft mask class 0")
            display_list.append(gt_soft)
        fig = plt.figure(figsize=(12, 9))

        for i in range(len(display_list)):
            img = display_list[i].numpy()
            plt.subplot(len(display_list)//2+1, 2, i+1)
            plt.title(title[i])
            plt.imshow(img)
            plt.axis('off')
        plt.show()
        if(upload_wandb):
            imgname = (str(counter) + "_" + str(outpath))
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                 sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            wandb.log({imgname: [wandb.Image(data, caption=imgname)]})
        if(not os.path.exists(outpath)):
            os.makedirs(outpath)
        plt.savefig(outpath + "visualization_" + str(datetime.now()) + ".png")


def debugvisualize(image, label, args, mattr):
    try:
      image = image.numpy()
      label = label.numpy()
    except:
      print("image was already of type ndarray")
    image = image.astype(np.uint8)
    if(args.classification):
        plt.imshow(image)
        label = label.numpy()
        label = str(label)
        # print("label is", label)
        plt.show()
    else:
        import cv2
        fig, axs = plt.subplots(3)

        label = np.reshape(label, (mattr.outheight, mattr.outwidth, mattr.nclasses))

        label_resized = cv2.resize(label, (mattr.inwidth, mattr.inheight), cv2.INTER_NEAREST)[:,:,0]
        label_resized = np.reshape(label_resized, (mattr.inheight, mattr.inwidth, 1))
        axs[0].imshow((image * ((1-label_resized))).astype(np.uint8))
        axs[1].imshow((image).astype(np.uint8))
        axs[2].imshow(label_resized[:,:,0])
        plt.show()
