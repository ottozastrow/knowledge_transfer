import argparse
import os
import shutil
import ast
from datetime import datetime

import numpy as np
import tensorflow as tf
# import matplotlib
# if('DISPLAY' not in os.environ.keys()):
#     matplotlib.use('agg')
# import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

import Models
import datasets
import visualization_tools
import teaching_tools
import DistillationMethods
import metrics_and_losses
import latency_tools


def get_custom_layers():
    """ in order to load model weights with custom layers
    these layers need to be passed to the loader"""
    import Models.mobilenetv3.layers as moblayers

    from Models.mobilenetv3.utils import LayerNamespaceWrapper
    from inspect import getmembers, isfunction
    from Models import layers

    custom_layers = dict([o for o in getmembers(layers) if isfunction(o[1])])
    custom_layers.update({
        'Bneck': moblayers.Bneck,
        'LayerNamespaceWrapper': LayerNamespaceWrapper,
        'ConvNormAct': moblayers.ConvNormAct,
        'BatchNormalization': moblayers.BatchNormalization,
        'Identity': moblayers.Identity,
        'ReLU6': moblayers.ReLU6,
        'SEBottleneck': moblayers.SEBottleneck,
        'LastStage': moblayers.LastStage,
        'HardSigmoid': moblayers.HardSigmoid,
        'HardSwish': moblayers.HardSwish,
        'Squeeze': moblayers.Squeeze,
        'GlobalAveragePooling2D': moblayers.GlobalAveragePooling2D,
    })
    custom_layers['hardswish'] = metrics_and_losses.hardswish

    return custom_layers


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="debug")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="enables visualization of "
                        "model output every 10 epochs during training")
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default="lawnbot")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_visualize", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--AT", action="store_true", default=False)
    parser.add_argument("--teacher_name", type=str, default="",
                        help="needed if using AT - model name"
                        "of teacher needs to be seet")

    parser.add_argument("--optimizer_name", type=str, default="rmsprop")
    parser.add_argument("--loss_name", type=str, default="miou")
    parser.add_argument("--image_normalization", type=str, default="sub_mean")

    parser.add_argument("--model_name", type=str, default="segnet_modular")
    parser.add_argument("--initialize_weights", type=str, default="",
                        help="set the model filepath that you want"
                        "to use as the initial weights")
    parser.add_argument("--debugdata", action="store_true", default=False)
    parser.add_argument("--classification", action="store_true", default=False)

    parser.add_argument("--BAN", action="store_true", default=False,
                        help="set to use sequence of self distillation")
    parser.add_argument("--RCO", action="store_true", default=False,
                        help="set to use sequence of "
                        "Route contrained Optimization")
    parser.add_argument("--teacher_weights_list", nargs='+',
                        help="only use with RCO - provide paths to teacher weights")
    parser.add_argument("--TA", action="store_true", default=False,
                        help="set to use sequence of Teacher Assistant Distillation")
    parser.add_argument("--model_names_list", nargs='+',
                        help="only use with TA - provide names of models")
    parser.add_argument("--model_options_list", nargs='+',
                        help="only use with TA - provide options of models")
    parser.add_argument("--oncpu", action="store_true", default=False)
    parser.add_argument("--delete_distill_dataset", action="store_true",
                        default=False)
    parser.add_argument("--zoom_intensity", type=float, default=0.25)
    parser.add_argument("--options", type=str, default="{}")
    parser.add_argument("--teacher", type=str, default="")
    parser.add_argument("--teacher_options", type=str, default="{}")
    parser.add_argument("--teacher_output", type=str, default="")
    parser.add_argument("--distillation", type=str, default="")
    parser.add_argument("--at_without_kd", action="store_true", default=False)
    parser.add_argument("--disable_tflite_inference", action="store_true",
                        default=False)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--temp", type=int, default=3)
    parser.add_argument("--subclasses_per_class", type=int, default=1)
    parser.add_argument("--subclass_beta", type=float, default=0.1,
                        help="weight of auxillary loss for subclass training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="rmsprop learning rates")
    parser.add_argument("--zero_kdloss_epoch", type=int, default=0,
                        help="KD loss will be set to zero at this epoch "
                        "during student training when using distillation")
    parser.add_argument("--measure_latency", action="store_true",
                        default=False)
    parser.add_argument("--augment", type=str2bool, nargs='?', const=True,
                        default=True)
    parser.add_argument("--connect_raspi", action="store_true", default=False)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--downblocks", type=int, default=5)
    parser.add_argument("--upblocks", type=int, default=2)
    parser.add_argument("--decoder", type=str, default="up_plain")
    parser.add_argument("--encoder", type=str, default="down_unet")
    parser.add_argument("--inheight", type=int, default=128)
    parser.add_argument("--inwidth", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--optimality_weight", type=float, default=-0.005)
    parser.add_argument("--target_latency", type=float, default=0.2)
    parser.add_argument("--double_convs_until", type=int, default=-1)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--keras_weights", type=str, default="imagenet")
    parser.add_argument("--keras_alpha", type=float, default=0.5,
                        help="only applicable for keras models. See keras doc")
    parser.add_argument("--channel_decoder_multiplier", type=int, default=-1)
    parser.add_argument("--padding", type=str, default="same",
                        help="enter same or valid")
    parser.add_argument("--input_ratio", type=float, default=2,
                        help="inheight * ratio = inwidth")
    parser.add_argument("--traintest_split", type=float, default=0.8,
                        help="proportion of train dataset to"
                        "total number of samples")
    parser.add_argument("--activation", type=str, default="relu")

    args = parser.parse_args()
    if(not args.disable_wandb):
        run = wandb.init(project="seg_practical", name=args.run_name,
                         group=args.experiment_name)
        wandb.config.update(args)
    else:
        run = None
    return args, run


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_teacher_batchsize(student_mattr):
    # compute a batch size that will result in batches which are ~25 MB in size
    floats_per_sample = student_mattr.inheight * student_mattr.inwidth * 3
    mb_per_sample = floats_per_sample * 32 / 4 / 1024 / 1024
    computed_batch_size = 25 // mb_per_sample
    print("computed batch size is ", computed_batch_size)
    return computed_batch_size


def train(args, run):
    custom_layers = get_custom_layers()
    calculated_inwidth = int(args.inheight * args.input_ratio)

    options = {'encoder': args.encoder,
               'decoder': args.decoder,
               'down_blocks': args.downblocks,
               'up_blocks': args.upblocks,
               'padding': args.padding,
               'dropout_rate': args.dropout_rate,
               'double_convs_until': args.double_convs_until,
               'channel_multiplier':
               (np.exp2(args.channel_multiplier)),
               'channel_decoder_multiplier':
               (np.exp2(args.channel_decoder_multiplier)),
               'keras_alpha': args.keras_alpha,
               'keras_weights': args.keras_weights,
               'activation': args.activation}
    options.update(ast.literal_eval(args.options))
    teacher_options = options

    BATCH_SIZE = args.batch_size
    if(args.dataset == "lawnbot"):
        nclasses = 2
    elif(args.dataset == "cityscapes"):
        nclasses = 30
    elif(args.dataset == "cifar10" or args.dataset == "mnist"):
        nclasses = 10
        assert(args.classification)
    elif(args.dataset == "barcodes"):
        nclasses = 2

    if not args.debug:
        EPOCHS = args.epochs
    else:  # make debug fast by using small amounts of data
        EPOCHS = 1
        BATCH_SIZE = 3

    modelFns = {
        'mobilenetv2': Models.Mobile_Net_v2.mobilenet,
        'mobilenetv2_unet': Models.mobilenetv2.unet_model,
        'mobilenetv2_classifier': Models.mobilenetv2_classifier.model,
        'shufflenetv2': Models.ShuffleNet.ShuffleNet,
        'shufflenet_hr': Models.shufflenet_hr.ShuffleNet,
        'xception_deeplab': Models.Xception_deeplab.Xception,
        'deeplab_test': Models.Deeplab_test.Deeplabv3,
        'segnet_dropout': Models.segnet_dropout.SegmentationModel,
        'segnet_modular': Models.segnet_modular.SegmentationModel,
        'segnet_dropout_512': Models.segnet_dropout_512.SegmentationModel,
        'segnet_classifier': Models.segnet_classifier.classification_model,
        'resnetv2': Models.resnetv2.classification_model,
        'resnet56': Models.resnetv2_56.make_model,
        'vggcifar': Models.vgg_cifar10.classification_model,
        'cifartiny': Models.vgg_cifar10_tiny.classification_model_tiny,
    }
    modelFN = modelFns[args.model_name]

    distill_fns = {
        '': DistillationMethods.plain.plain_training,
        'hinton_distillation': DistillationMethods.hinton_distillation.distill
    }
    distill_fn = distill_fns[args.distillation]

    load_dataset_fns = {
        'cityscapes': datasets.load_cityscapes,
        'lawnbot': datasets.load_lawnbot,
        'cifar10': datasets.load_cifar10,
        'mnist': datasets.load_mnist,
        'barcodes': datasets.load_barcodes,
        'barcodes_synth': datasets.load_barcodes_synth,
    }
    load_dataset_fn = load_dataset_fns[args.dataset]
    # subclass KD requires loss to be computed from logits
    from_logits = (args.subclasses_per_class != 1)

    mattr = teaching_tools.ModelAttribs(
        nclasses=nclasses,
        batch_size=BATCH_SIZE, from_logits=from_logits,
        initialize=args.initialize_weights,
        model_name=args.model_name, model_options=options,
        classification=args.classification,
        inheight=args.inheight, inwidth=calculated_inwidth)
    print("finished creating mattr")

    teacher_batchsize = compute_teacher_batchsize(student_mattr=mattr)
    teacher_mattr = teaching_tools.ModelAttribs(
        nclasses=nclasses,
        batch_size=teacher_batchsize, from_logits=from_logits,
        initialize=args.teacher, model_name=args.teacher_name,
        output=args.teacher_output, model_options=teacher_options,
        classification=args.classification,
        inheight=args.inheight, inwidth=calculated_inwidth)

    # get output dimensions for modularnets
    if(args.model_name == "segnet_modular"):
        # in segnet modular the output dimensions are determined by the model
        # config. the easiest way to get the dimensions is to create a model
        tmpmodel = modelFN(mattr, **(mattr.model_options))
        mattr.outheight = tmpmodel.outheight
        mattr.outwidth = tmpmodel.outwidth
        del tmpmodel
        if args.distillation == "hinton_distillation":
            tmpmodel = modelFN(teacher_mattr, **(teacher_mattr.model_options))
            teacher_mattr.outheight = tmpmodel.outheight
            teacher_mattr.outwidth = tmpmodel.outwidth
            del tmpmodel

    load_ds = not (args.distillation == "hinton_distillation"
                   and args.teacher_output != "")
    if(load_ds):
        train_dataset, test_dataset, TRAIN_LENGTH, VAL_LENGTH, _ =\
            load_dataset_fn(mattr, args)
    else:
        train_dataset = None
        test_dataset = None
        TRAIN_LENGTH, VAL_LENGTH = datasets.dataset_details(args.dataset)

    if(args.debug):
        TRAIN_LENGTH = 6
        VAL_LENGTH = 6

    TRAIN_STEPS = TRAIN_LENGTH // BATCH_SIZE
    VAL_STEPS = VAL_LENGTH // BATCH_SIZE
    assert(VAL_STEPS != 0), [VAL_LENGTH, BATCH_SIZE]
    assert(TRAIN_STEPS != 0)

    if(load_ds):
        train_dataset, test_dataset = datasets.prepare_dataset(
            train_dataset, test_dataset, TRAIN_LENGTH, mattr,
            args, records_from_teacher=False)

    # for subclass distillation the number of classes
    # for the teacher and student model is changed.
    mattr.nclasses *= args.subclasses_per_class
    teacher_mattr.nclasses *= args.subclasses_per_class

    logs = ("logs/" + args.experiment_name + "/run_" + args.run_name
            + "_" + str(datetime.now()) + "/")

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='10,12')

    checkpoints_path = (
      "checkpoint_weights/" + args.experiment_name
      + "/run_" + args.run_name + "_" + str(datetime.now()) + "/")

    callbacks_list = [teaching_tools.bestModelCallback(
                        checkpoints_path, args=args, run=run),
                      teaching_tools.logParamsCount(args), tboard_callback]

    if(not args.disable_wandb):
        callbacks_list.append(WandbCallback())

    if(args.visualize):
        callbacks_list.append(
            visualization_tools.DisplayCallback(mattr, args, train_dataset))

    stages = args.stages

    if(args.RCO):
        teacher_weights_list = args.teacher_weights_list

        stages = len(teacher_weights_list)
        assert(stages > 0)
        for el in teacher_weights_list:
            if(el != ""):
                assert(os.path.exists(el))

    if(args.TA):
        model_names_list = args.model_names_list
        assert(args.model_name == model_names_list[0])
        stages = len(model_names_list)
        model_options_list = args.model_options_list
        assert(len(model_options_list) == len(model_names_list))

    teacher_model = None
    model = None
    combined_epoch_counter = 0

    teacher_is_next_student = args.BAN or args.TA

    if(args.BAN):
        assert(args.teacher_name == args.model_name)
        assert(args.teacher_options == options)

    if(not (args.RCO or args.BAN or args.TA)):
        assert(stages <= 1)
    assert(args.RCO + args.BAN + args.TA <= 1)
    teacher_output_paths = []
    if(not args.debugdata):
        for stage in range(stages):
            if(args.RCO):
                teacher_model = None
                teacher_mattr = teaching_tools.ModelAttribs(
                    nclasses=nclasses,
                    batch_size=BATCH_SIZE,
                    from_logits=from_logits,
                    initialize=teacher_weights_list[stage],
                    model_name=args.teacher_name,
                    model_options=args.teacher_options,
                    classification=args.classification,
                    inheight=args.inheight, inwidth=calculated_inwidth)

            if(args.TA):  # using different teachers (-> not self distillation)
                mattr = teaching_tools.ModelAttribs(
                            nclasses=nclasses, batch_size=BATCH_SIZE,
                            from_logits=from_logits,
                            initialize="",
                            model_name=model_names_list[stage],
                            model_options=model_options_list[stage],
                            classification=args.classification,
                            inheight=args.inheight, inwidth=calculated_inwidth)

            model, train_dataset, test_dataset, teacher_output_path = distill_fn(
                args, mattr, teacher_mattr, teacher_model,
                model, custom_layers,
                VAL_LENGTH, VAL_STEPS, EPOCHS, TRAIN_STEPS, TRAIN_LENGTH,
                combined_epoch_counter, callbacks_list, test_dataset,
                train_dataset, modelFns, checkpoints_path)
            combined_epoch_counter += EPOCHS
            teacher_output_paths.append(teacher_output_path)

            if(teacher_is_next_student):
                teacher_model = model
                model = None
                teacher_mattr = mattr
                teacher_mattr.batch_size = compute_teacher_batchsize(
                    student_mattr=mattr)

                mattr = teaching_tools.ModelAttribs(
                    nclasses=nclasses,
                    batch_size=BATCH_SIZE, from_logits=from_logits,
                    initialize="", model_name=args.model_name,
                    model_options=options, classification=args.classification,
                    inheight=args.inheight, inwidth=calculated_inwidth)

        if(args.evaluate):
            print("evaluating test_dataset")
            model.evaluate(test_dataset, steps=VAL_STEPS, verbose=2)

        if(args.predict):
            print("making predictions for test_dataset")
            visualization_tools.show_predictions(
                mattr, dataset=test_dataset, args=args, model=model)

        if(args.measure_latency):
            latency = visualization_tools.measure_latency(
                model, test_dataset, args.disable_tflite_inference,
                batch_size=BATCH_SIZE)
            print("average inference latency: ", latency)
            if(not args.disable_wandb):
                wandb.log({"latency": latency})

        # created datasets take up a lot of space
        if(args.delete_distill_dataset):
            for path in teacher_output_paths:
                if(path != None):
                    print("deleting teacher output path at:", path)
                    shutil.rmtree(path)

    return model, test_dataset, BATCH_SIZE, mattr


if __name__=="__main__":
    args, run = get_args()
    if(args.connect_raspi):
        latency_tools.call_raspberrypi(vars(args), run)
    train(args, run)
