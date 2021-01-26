import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def hardswish(x):
    """ hard swish activation function from
    searching for MobileNetv3 paper"""

    return x * tf.nn.relu6(x+3) * 0.16666667


def miou_score():
    def miou_score(gt, pr):

        axes = [0, 1, 2]

        intersection = K.sum(gt * pr, axis=axes)
        union = K.sum(gt + pr, axis=axes) - intersection
        # iou = (intersection + 0.01) / (union + 0.01)
        iou = (intersection + 1) / (union + 1)
        miou = K.mean(iou)

        return miou
    return miou_score


def miou_loss():
    def miou_loss(gt, pr):

        axes = [0, 1, 2]

        intersection = K.sum(gt * pr, axis=axes)
        union = K.sum(gt + pr, axis=axes) - intersection
        # iou = (intersection + 0.01) / (union + 0.01)
        iou = (intersection + 1) / (union + 1)
        miou = K.mean(iou)

        return 1-miou
    return miou_loss


def miou_loss_flex(from_logits=False, temp=1.0):
    axes = [0, 1, 2]
    def miou_loss(gt, pr):
        if from_logits:
            pr = K.softmax(pr)

        intersection = K.sum(gt * pr, axis=axes)
        union = K.sum(gt + pr, axis=axes) - intersection
        # iou = (intersection + 0.01) / (union + 0.01)
        iou = (intersection + 1) / (union + 1)
        miou = K.mean(iou)

        return (1-miou) * K.square(float(temp))
    return miou_loss


def miou_score_subclasses(subclasses_per_class, from_logits=False):

    def miou_score(gt, pr):
        batch_size = tf.shape(pr)[0]
        if(from_logits):
            pr = K.softmax(pr)
        if(subclasses_per_class == 1):
            pr = tf.reshape(pr, (batch_size, tf.shape(pr)[1], tf.shape(pr)[2]))
        else:
            orig_classes = tf.shape(pr)[-1]//subclasses_per_class
            pr = tf.reshape(pr, (batch_size, tf.shape(pr)[1],
                                 orig_classes, subclasses_per_class))

            pr = K.sum(pr, axis=-1, keepdims=False)
        axes = [0, 1, 2]
        intersection = K.sum(gt * pr, axis=axes)
        union = K.sum(gt + pr, axis=axes) - intersection
        # iou = (intersection + 0.01) / (union + 0.01)
        iou = (intersection + 1) / (union + 1)
        miou = K.mean(iou)

        return miou
    return miou_score


def at_loss(args):
    def loss(teachermap, studentmap):
        # teachermap = tf.reshape(teachermap, (-1))
        #interpolation for adjusting difference in size
        # #(also because during transfer the at map was stored at input image size)
        # teachermap = tf.reshape(teachermap, (studentmap_size[0], studentmap_size[1], 1))
        # teachermap = tf.image.resize(teachermap, (studentmap_size[0], studentmap_size[1]))
        # teachermap = tf.reshape(teachermap, (studentmap_size[0], studentmap_size[1]))

        teachermap = K.flatten(teachermap)
        studentmap = K.flatten(studentmap)
        # studentmap = tf.reshape(studentmap, (-1))
        teachermap = tf.math.l2_normalize(teachermap)
        studentmap = tf.math.l2_normalize(studentmap)

        distance = studentmap - teachermap
        distance = tf.math.square(distance)

        distance = tf.math.reduce_sum(distance, axis=-1)
        distance = tf.math.sqrt(distance)
        #distance =  1 / 2 * distance
        return distance
    return loss


def subclass_acc(subclasses_per_class):
    fn = tf.keras.metrics.CategoricalAccuracy()
    def acc(ytrue, pred):
        pred = K.softmax(pred)

        orig_classes = tf.shape(pred)[-1]//subclasses_per_class
        # orig_classes = 2
        #pred = tf.reshape(pred, (tf.shape(pred)[0], tf.shape(pred)[1], orig_classes, subclasses_per_class))
        pred = tf.reshape(pred, (tf.shape(pred)[0], orig_classes, subclasses_per_class))

        #sum_reduce subclasses
        pred = K.sum(pred, axis=-1, keepdims=False)

        return fn(ytrue, pred)
    return acc


def crossentropy_subclasses(temp, subclasses_per_class, from_logits, beta,
                            args=None):
    def apply_softmax(pred, T):
        dividend = tf.math.exp(pred/T)
        divisor = tf.reduce_sum(tf.math.exp(pred/T), axis=-1)
        divisor = tf.expand_dims(divisor, axis=-1)
        return tf.math.multiply(1/divisor, dividend)
        # return K.softmax(pred/T)

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred)
        loss = -K.sum(loss, -1)

        return loss * K.square(float(temp))

    def loss_cross_aux(y_true, y_pred_aux):
        def cross(y_true, pred):
            if(subclasses_per_class != 1):
                orig_classes = tf.shape(pred)[-1] // subclasses_per_class
                # orig_classes = 2
                hw = tf.shape(pred)[1]
                batchsize = tf.shape(pred)[0]

                # pred = tf.reshape(pred, (tf.shape(pred)[0],
                #     tf.shape(pred)[1], orig_classes, subclasses_per_class))
                pred = tf.reshape(pred, (batchsize, hw,
                                         orig_classes, subclasses_per_class))

                # sum_reduce subclasses
                pred = K.sum(pred, axis=-1, keepdims=False)

            # pred = apply_softmax(pred, temp)
            pred = K.softmax(pred/temp)

            loss_cross_entropy = loss(y_true, pred)

            return loss_cross_entropy

        def aux(y_true, y_pred):
            # compute loss aux
            predshape = tf.cast(tf.shape(y_pred), tf.float32)
            hw = predshape[1]
            batch_size = predshape[0]
            nclasses = predshape[-1]

            logits = K.sum(y_pred, axis=1) / hw  # collapse hw dimmension
            # logits = y_pred
            logits_normalized = (logits-K.mean(logits))  # / K.std(logits)
            range_min_max = tf.reduce_max(logits_normalized)\
                - tf.reduce_min(logits_normalized)

            logits_normalized = logits_normalized/range_min_max

            # logits_normalized = tf.transpose(logits_normalized, perm=[1, 0, 2])
            # #hw x class x batch (.) batch class hw. Orig = #batch hw class
            # a = tf.transpose(logits_normalized, perm=[1, 2, 0])
            # b = tf.transpose(logits_normalized, perm=[])
            #dot = tf.tensordot(logits_normalized,logits_normalized, axes=((0), (0))) / temp
            # dot = K.dot(, logits_normalized) / temp
            #print("dot shape", dot.shape)
            # print(dot)

            #####only classification
            logits_normalized = tf.reshape(logits_normalized, (batch_size,nclasses))

            transposed_logits_normalized = tf.transpose(logits_normalized)

            #dot = tf.tensordot(transposed_logits_normalized, logits_normalized, axes=0) / temp
            dot = tf.linalg.matmul(logits_normalized, transposed_logits_normalized) / temp
            ######
            inner_bracket = (K.sum(K.exp(dot), axis=-1))
            #tf.print("summands: ", inner_bracket)
            summand1 = K.sum(K.log(inner_bracket)) / batch_size
            summand2 = 0#-1/temp - K.log((batch_size))

            loss_aux = summand1 + summand2
            #tf.print(loss_aux)
            #print("loss aux", loss_aux, tf.shape(loss_aux))
            return loss_aux

        return (1-beta)*cross(y_true, y_pred_aux) + beta*aux(
            y_true, y_pred_aux)# * K.square(float(temp))
        # return (cross(y_true, y_pred_aux) * 0 + beta*aux(
        #     y_true, y_pred_aux)) * K.square(float(temp)) #this line is temporary, use above

    if(not from_logits):
        return loss
    else:
        return loss_cross_aux


def miou_score_student_hard():
    def miou_score_student_hard(gt, pr):

        #because output of student model is shape: [1, width*height, n_classes*2] we have to change it

        pr_hard = pr[:,:,:2] #use softmax without temperature vs with : pr[0,:,2:]
        gt_hard = gt[:,:,:2] #use hard ground_truth

        pr_hard = K.argmax(pr_hard, axis=2)
        gt_hard = K.argmax(gt_hard, axis=2)

        intersection = K.sum(gt_hard * pr_hard)
        union = K.sum(gt_hard + pr_hard) - intersection
        iou = (intersection + 1) / (union + 1)
        miou = K.mean(iou)

        return miou
    return miou_score_student_hard

