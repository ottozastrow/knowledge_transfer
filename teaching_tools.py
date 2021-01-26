import tensorflow as tf
import preprocessing_tf
import math
import os
import wandb
import visualization_tools
import numpy as np



def predict_and_save_tfrecords(teacher_input_ds, teacher_model, filepath, STEPS,
        mattr, student_mattr, args):

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy().tobytes() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def serialize_example(image, gt, pred):
        feature = {
            'image': _bytes_feature(image),
            'gt': _bytes_feature(gt),
            'pred': _bytes_feature(pred),
            'inheight': _int64_feature(mattr.inheight),
            'inwidth': _int64_feature(mattr.inwidth),
            'outheight': _int64_feature(mattr.outheight),
            'outwidth': _int64_feature(mattr.outwidth),
            'nclasses': _int64_feature(mattr.nclasses),
        }
        #  Create a Features message using tf.train.Example.
        return  tf.train.Example(features=tf.train.Features(feature=feature))

    def tf_serialize_example(f0,f1,f2):
        tf_string = tf.py_function(
            serialize_example,
            (f0,f1,f2),  # pass these args to the above function.
            tf.string)      # the return type is `tf.string`.
        return tf_string # The result is a scalar

    #because of a bug in tf.data.Datasets the samples are not in a tf.example format
    #instead resized and concatenated to a single tensor
    def concat3to1(img, gt, prediction):
        # bigsize = (img.shape[0], img.shape[1])
        bigsize = (mattr.inheight, mattr.inwidth)

        gt = tf.reshape(gt, (student_mattr.outheight, student_mattr.outwidth, mattr.nclasses//args.subclasses_per_class))
        gt = tf.image.resize(gt, (bigsize), tf.image.ResizeMethod.BILINEAR)
            
        prediction = tf.reshape(prediction, (mattr.outheight, mattr.outwidth, mattr.nclasses))
        prediction = tf.image.resize(prediction, (bigsize), tf.image.ResizeMethod.BILINEAR)

        concatenated = tf.concat([img, gt, prediction], axis=-1)

        return concatenated

    def concatAttention(img, gt, prediction, attention_map):
        bigsize = (mattr.inheight, mattr.inwidth)
        attention_map_size = tf.shape(attention_map)
        
        attention_map = tf.reshape(attention_map, (attention_map_size[0], attention_map_size[1], 1))
        # attention_map = tf.reshape(attention_map, (mattr.outheight, mattr.outwidth, 1))
        attention_map = tf.image.resize(attention_map, bigsize, tf.image.ResizeMethod.BILINEAR)
        
        img_and_attention = tf.concat([img, attention_map], axis=-1)
        return img_and_attention, gt, prediction

    if(not os.path.exists(filepath)):
        os.makedirs(filepath)
    counter = 0

    num_batches = math.ceil(STEPS/mattr.batch_size)
    for image_batch, gt_batch in teacher_input_ds.take(num_batches):
        pred_batch = teacher_model.predict(image_batch, steps=tf.constant(1))

        image_batch = tf.cast(preprocessing_tf.denormalize_sub_mean(
            image_batch, args.dataset), tf.float32)
        if(args.AT):
            attention_map_batch = pred_batch[1]
            pred_batch = pred_batch[0]

        two_outputs = tf.rank(pred_batch) == 4 and tf.shape(pred_batch)[0] == 2


        if(two_outputs):  # multiple outputs could come from a
            # student being used as teacher.
            pred_batch = pred_batch[0]
            print("two outputs")
            assert(False)

        slices_tuple = (image_batch, gt_batch, pred_batch)
        if(args.AT):
            slices_tuple = (image_batch, gt_batch,
                            pred_batch, attention_map_batch)

        feature_batch = tf.data.Dataset.from_tensor_slices(slices_tuple)

        if(args.AT):
            feature_batch = feature_batch.map(concatAttention)

        feature_batch = feature_batch.map(concat3to1)
        serialized_batch = feature_batch.map(tf.io.serialize_tensor)

        filename = filepath + "serialized_batch_" + str(counter) + ".tfrecord"
        counter += 1

        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_batch)

        print("saved ", filename)


def replace_gt_with_softlabels(img, masks):
    # masks is a tuple of [mask_previous_teacher, mask_current_teacher]
    return img, masks[0]


def add_temp_output(model, temp, apply_softmax, mattr):
    # this is only for the student - probabilities_T is only to
    # compute the loss with the teachers probabilites_T output
    # probabilites is the actual output and is also used for the gt loss
    # print(model.summary())
    # print("this was before add temp output")
    if(mattr.from_logits):
        logits = model.layers[-1].output
    else:
        # if a model has a softmax ergo the model is not from_logits,
        # the logits is the second last layer
        logits = model.layers[-2].output

    layer_names = [layer.name for layer in model.layers]

    # neceseary because of unexpected behaviour
    if("reshape" not in layer_names):
        logits = tf.keras.layers.Reshape(
            (mattr.outheight*mattr.outwidth, mattr.nclasses))(logits)

    probabilities = tf.keras.layers.Activation(
        'softmax', name="output")(logits)

    logits_T = tf.keras.layers.Lambda(
        lambda x, temp: x / temp, arguments={'temp': temp},
        name="temp_logits")((logits))
    probabilities_T = tf.keras.layers.Activation(
        'softmax', name="activation_temp")(logits_T)

    mnew = tf.keras.models.Model(
        inputs=model.input, outputs=[probabilities, probabilities_T])

    model = mnew
    return model


def get_sizes_for_modelname(modelname, classification):
    inheight, inwidth = 128, 256
    outheight, outwidth = 128, 256
    if(classification):
        inheight, inwidth = 32, 32
        outheight, outwidth = 1, 1
    if(modelname == "segnet_dropout"):
        outheight, outwidth = 16, 32

    return inheight, inwidth, outheight, outwidth


def turn_on_the_heat(model, temp, mattr):
    # this is only for the teacher - before passing on knowledge to the
    # student the output is modified to include a temperature

    if(mattr.from_logits):
        logits = model.layers[-1].output
    else:
        # if a model has a softmax ergo the model is not from_logits,
        # the logits is the second last layer
        logits = model.layers[-2].output

    logits_T = tf.keras.layers.Lambda(
        lambda x, temp: x / temp, arguments={'temp': temp},
        name="temp_logits")((logits))
    # neceseary because of unexpected behaviour
    logits = tf.keras.layers.Reshape(
        (mattr.outheight*mattr.outwidth, mattr.nclasses),
        name="reshapeoutput")(logits)

    probabilities_T = tf.keras.layers.Activation(
        'softmax', name="activation_temp")(logits_T)
    mnew = tf.keras.models.Model(
        inputs=model.input, outputs=[probabilities_T])
    model = mnew
    # print(model.summary())
    # print("the above model was given increased temperature")
    return model


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(AttentionLayer, self).__init__(name=name)

    def build(self, input_shape):
        pass

    def call(self, input):
        x = tf.math.abs(input.output)
        x = tf.math.square(x)
        x = tf.math.reduce_sum(x, axis=-1)
        return x


def add_attention_map(model, mattr):
    attention_layers_config = {
        'xception_deeplab': ['decoder_conv1_depthwise_BN'],
        # 'shufflenet_hr':['decoderconv2d_3'],
        'shufflenet_hr': ['stage2/block2/bn_1x1conv_1'],
        'segnet_dropout': ['batchnorm_attach_attention'],
    }

    attentionlayername = attention_layers_config[mattr.model_name][0]
    connection_layer = model.get_layer(attentionlayername)
    attention_map = AttentionLayer(name="attention_map")(connection_layer)
    outputs = model.outputs
    outputs.append(attention_map)

    model = tf.keras.models.Model(inputs = model.input, outputs=outputs)
    return model


def make_model_single_output(source_model):
    model=tf.keras.models.Model(
        inputs = source_model.inputs, outputs = source_model.outputs[0])
    return model


class logParamsCount(tf.keras.callbacks.Callback):
    def __init__(self, args):
        self.args = args

    def on_train_begin(self, logs=None):
        if(not self.args.disable_wandb):
            wandb.run.summary.update(
                {"trainable params:": self.model.count_params()})
        print("number of trainable params for model is:",
              self.model.count_params())


class bestModelCallback(tf.keras.callbacks.Callback):

    current_best_measure = -0.1
    current_best_epoch = 0

    def __init__(self, dirname, args, run):
        self.dirname = dirname + "/"
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.args = args
        self.lastepoch = -1
        self.run = run

    def on_epoch_end(self, epoch, logs={}):
        # epoch actually is step here. keras bug?
        measure = 0
        possible_metrics = ["val_output_miou_score",
                            "val_output_miou_score_student_hard",
                            "val_miou_score",
                            "val_miou_score_student_hard",
                            "val_acc",
                            "val_output_acc"]
        chosen_key = ""
        if(self.lastepoch < epoch):
            self.lastepoch = epoch
            if(not self.args.disable_wandb):
                wandb.log({'best_val_miou': self.current_best_measure,
                           'epoch': epoch})
                wandb.log({'best_epoch': self.current_best_epoch})
                if 'latency' in self.run.summary.keys():
                    # optimality is for the parameter to be
                    # maximized by hyper param search
                    optimality = self.current_best_measure * (
                        ((self.run.summary['latency'] + 0.02) / self.args.target_latency) **
                        self.args.optimality_weight)
                    self.run.summary['optimality'] = optimality
                    self.run.summary.update()
                    print("\n\n")
                    print("optimality is ", optimality)
                    print("\n\n")

        for key in possible_metrics:
            if key in logs.keys():
                measure = logs[key]
                chosen_key = key

                if(measure > self.current_best_measure):
                    print(" - measure is ", measure, " ")
                    self.current_best_epoch = epoch

                    self.current_best_measure = measure

                    filename = "model-" + str(epoch) + "-" + str(measure)[:6]\
                        + "-" + chosen_key + ".h5"
                    print("saving new best model with name: ",
                          filename, " in ", self.dirname)
                    self.model.save(self.dirname + filename,
                                    include_optimizer=False)
                    print("saving model at ", self.dirname)
                return
        print("not saving model - no metric detected. Metric found was",
              logs.keys())


class ModelAttribs:
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.nclasses = kwargs['nclasses']
        self.from_logits = kwargs['from_logits']
        self.model_options = kwargs['model_options']
        self.model_name = kwargs['model_name']
        self.initialize = kwargs['initialize']
        self.inwidth = kwargs['inwidth']
        self.inheight = kwargs['inheight']
        self.outheight = 256  # 16
        self.outwidth = 256  # 32

        if('output' in kwargs.keys()):
            # output is a path to a directory of tfrecords.
            # These records a the output from a
            # trained teacher and can be reused by setting this variable
            self.output = kwargs['output']
        else:
            self.output = ""

        # sizes = get_sizes_for_modelname(self.model_name,
        #       classification=kwargs['classification'])
        # self.inheight, self.inwidth, self.outheight, self.outwidth  = sizes
    def __str__(self):
        return str("model_name: ", self.model_name, " input h x w: ",
                   self.inheight, self.inwidth, " output h x w: ",
                   self.outheight, self.outwidth, " batchsize: ",
                   self.batch_size, " model_options: ", self.model_options)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
