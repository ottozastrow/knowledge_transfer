import tensorflow as tf
import metrics_and_losses
import json
from datetime import datetime
import teaching_tools
import datasets


def distill(args, student_mattr, teacher_mattr, teacher_model, student,
            custom_layers, VAL_LENGTH, VAL_STEPS, EPOCHS,
            TRAIN_STEPS, TRAIN_LENGTH, combined_epoch_counter,
            callbacks_list, test_dataset, train_dataset,
            modelFns, checkpoints_path):

    assert(teacher_mattr.model_name != ""),\
        "teacher_name must be set when using distillation"
    print()
    print("Distilling Knowledge from ", teacher_mattr.model_name,
          teacher_mattr.model_options,
          " to", student_mattr.model_name, student_mattr.model_options)
    print("new teacher model created/initialized:", teacher_model is None,
          ", teacher weights path:", teacher_mattr.initialize)
    print("new student model created/initialized:", student is None,
          ", student weights path:", student_mattr.initialize)
    print("teacher output:", teacher_mattr.output)
    print()
    custom_objects = {}
    custom_objects.update(custom_layers)
    if(teacher_mattr.output == ""):
        if(not args.classification):
            custom_objects['miou_score'] = \
                metrics_and_losses.miou_score_subclasses

        if(teacher_model is None):
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                if(teacher_mattr.initialize == ""):
                    modelFN = modelFns[teacher_mattr.model_name]
                    teacher_model = modelFN(
                        teacher_mattr, **((teacher_mattr.model_options)))
                    print("debug dummmy teacher generated")
                else:
                    teacher_model = tf.keras.models.load_model(
                        teacher_mattr.initialize, compile=False,
                        custom_objects=custom_objects)
        teacher_model = teaching_tools.turn_on_the_heat(
            teacher_model, args.temp, teacher_mattr)

        if(args.AT):
            teacher_model = teaching_tools.add_attention_map(teacher_model,
                                                             teacher_mattr)
        print(teacher_model.summary())
        print("above is teacher")

        teacher_output_path = "teacher_output/" + args.experiment_name +\
            "/run_" + args.run_name + "_" + str(datetime.now()) + "/"
        teaching_tools.predict_and_save_tfrecords(
            test_dataset, teacher_model, teacher_output_path+"test/",
            VAL_LENGTH, teacher_mattr, student_mattr, args=args)
        teaching_tools.predict_and_save_tfrecords(
            train_dataset, teacher_model, teacher_output_path+"train/",
            TRAIN_LENGTH, teacher_mattr, student_mattr, args=args)
    else:
        teacher_output_path = teacher_mattr.output + "/"
    tf.keras.backend.clear_session()

    train_dataset, test_dataset = datasets.load_teacher(
        teacher_output_path, student_mattr, args)

    train_dataset, test_dataset = datasets.prepare_dataset(
        train_dataset, test_dataset, TRAIN_LENGTH, student_mattr,
        args, records_from_teacher=True)

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    if(True):
        if(args.classification):
            metrics = ["acc", tf.keras.metrics.KLDivergence()]
            # losses=["categorical_crossentropy"]
            custom_objects = {}
        else:
            metrics = {'output': metrics_and_losses.miou_score_subclasses(
                subclasses_per_class=args.subclasses_per_class),
                        'activation_temp': [metrics_and_losses.miou_score()]}

            custom_objects = {'miou_score':
                              metrics_and_losses.miou_score_subclasses}
        losses = {}
        if args.loss_name == "miou":
            losses['output'] = metrics_and_losses.miou_loss_flex(
                from_logits=student_mattr.from_logits, temp=1)
            losses['output'] = metrics_and_losses.miou_loss_flex(
                from_logits=student_mattr.from_logits, temp=args.temp)
        else:
            losses['output'] = metrics_and_losses.crossentropy_subclasses(
                temp=1, subclasses_per_class=args.subclasses_per_class,
                from_logits=student_mattr.from_logits,
                beta=args.subclass_beta, args=args)
            losses['activation_temp'] = metrics_and_losses.crossentropy_subclasses(
                temp=args.temp, from_logits=student_mattr.from_logits,
                beta=args.subclass_beta, subclasses_per_class=1, args=args)

        if(args.AT):
            losses['tf_op_layer_Sum'] = metrics_and_losses.at_loss(args)

        # optimizer = args.optimizer_name
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=args.learning_rate)
        modelFN = modelFns[student_mattr.model_name]

        if(student is None):
            student = modelFN(student_mattr, **((student_mattr.model_options)))

        # elif(student==None):
        #     student = tf.keras.models.load_model(
        #     args.initialize_weights, compile = False,
        #         custom_objects=custom_objects)

        student = teaching_tools.add_temp_output(
            student, args.temp, apply_softmax=not student_mattr.from_logits,
            mattr=student_mattr)
        if(args.AT):
            student = teaching_tools.add_attention_map(student, student_mattr)

        student.compile(optimizer=optimizer, loss=losses, metrics=metrics)

        print(student.summary())
        print("the above summary is the student")
        fit_epochs = combined_epoch_counter + EPOCHS
        if(args.zero_kdloss_epoch != 0):
            fit_epochs = combined_epoch_counter + args.zero_kdloss_epoch

        model_history = student.fit(train_dataset, epochs=fit_epochs,
                                    steps_per_epoch=TRAIN_STEPS,
                                    validation_steps=VAL_STEPS,
                                    validation_data=test_dataset,
                                    callbacks=callbacks_list,
                                    initial_epoch=combined_epoch_counter,
                                    verbose=2)

        if(args.zero_kdloss_epoch != 0):
            # setting the kd loss to 0 after some epochs of
            # training the students helps
            # here a trick is used: setting the kdloss temp parameter to
            # zero will multiply the gradient by 0
            losses['activation_temp'] =\
                metrics_and_losses.crossentropy_subclasses(
                    temp=0,
                    from_logits=student_mattr.from_logits,
                    beta=args.subclass_beta, subclasses_per_class=1, args=args)
            print("setting kdloss to zero from this epoch")
            student.compile(optimizer=optimizer, loss=losses, metrics=metrics)

            remaining_epochs = EPOCHS - args.zero_kdloss_epoch
            model_history = student.fit(
                train_dataset, epochs=remaining_epochs,
                initial_epoch=combined_epoch_counter+args.zero_kdloss_epoch,
                steps_per_epoch=TRAIN_STEPS,
                validation_steps=VAL_STEPS,
                validation_data=test_dataset,
                callbacks=callbacks_list,
                verbose=2)

        next_teacher = teaching_tools.make_model_single_output(student)

    test_dataset = test_dataset.map(teaching_tools.replace_gt_with_softlabels)
    train_dataset = train_dataset.map(
        teaching_tools.replace_gt_with_softlabels)
    return next_teacher, train_dataset, test_dataset, teacher_output_path
