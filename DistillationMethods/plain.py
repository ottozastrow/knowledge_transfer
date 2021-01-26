from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import wandb
import tensorflow as tf

import metrics_and_losses
import teaching_tools


def plain_training(args, mattr, teacher_mattr, teacher_model, student,
                   custom_layers, VAL_LENGTH, VAL_STEPS, EPOCHS,
                   TRAIN_STEPS, TRAIN_LENGTH, combined_epoch_counter,
                   callbacks_list, test_dataset, train_dataset,
                   modelFns, checkpoints_path):

    if(not args.classification):
        metrics = [metrics_and_losses.miou_score_subclasses(
            subclasses_per_class=args.subclasses_per_class,
            from_logits=mattr.from_logits),
            # metrics_and_losses.crossentropy_subclasses(
            #    temp=1, from_logits=mattr.from_logits,
            #    subclasses_per_class=args.subclasses_per_class,
            #    beta=0.0, args=args),
            metrics_and_losses.crossentropy_subclasses(
                temp=3, from_logits=mattr.from_logits,
                subclasses_per_class=args.subclasses_per_class,
                beta=1.0, args=args)]

        custom_objects = {'miou_score':
                          metrics_and_losses.miou_score_subclasses,
                          'miou_score_student_hard':
                          metrics_and_losses.miou_score_student_hard}
    else:
        metrics = []
        if(args.subclasses_per_class == 1):
            metrics = ["acc"]
        else:
            metrics.append(metrics_and_losses.subclass_acc(
                subclasses_per_class=args.subclasses_per_class))
            metrics.append(metrics_and_losses.crossentropy_subclasses(
                temp=1, from_logits=mattr.from_logits,
                subclasses_per_class=args.subclasses_per_class,
                beta=1.0, args=args))
        custom_objects = {}
    custom_objects.update(custom_layers)
    losses = [metrics_and_losses.crossentropy_subclasses(
        temp=1, from_logits=mattr.from_logits,
        subclasses_per_class=args.subclasses_per_class,
        beta=args.subclass_beta, args=args)]
    if(args.loss_name == "miou"):
        losses = [metrics_and_losses.miou_loss()]

    optimizer = args.optimizer_name
    if(args.optimizer_name == "scheduler"):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=teaching_tools.lr_schedule(0))

    def inner_load_and_compile():
        if(mattr.initialize == ""):
            modelFN = modelFns[mattr.model_name]
            model = modelFN(mattr, **(mattr.model_options))
        else:
            model = tf.keras.models.load_model(
                mattr.initialize, compile=False, custom_objects=custom_objects)
        # model = teaching_tools.make_model_single_output(model)

        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        return model

    if(not args.oncpu):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = inner_load_and_compile()
    else:
        model = inner_load_and_compile()

    print(model.summary())
    verbosity = 1
    if(args.oncpu):
        verbosity = 1

    model.fit(train_dataset, epochs=EPOCHS,
              steps_per_epoch=TRAIN_STEPS,
              validation_steps=VAL_STEPS,
              validation_data=test_dataset,
              callbacks=callbacks_list,
              verbose=verbosity,
              initial_epoch=combined_epoch_counter)

    return model, train_dataset, test_dataset, None
