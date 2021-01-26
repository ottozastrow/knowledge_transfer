import LoadBatches
import metrics_and_losses
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="weights//quantization_test/vgg_segnet_hacked_channelslast__id9108/weights-improvement-01-0.6931-0.33335140347480774.hdf5")

parser.add_argument("--train_images", type = str, default="../dataset_current/dataset_own_v8.3/images_train/"  )
parser.add_argument("--train_annotations", type = str , default="../dataset_current/dataset_own_v8.3/images_train/" )

parser.add_argument("--n_classes", type=int , default=2)
parser.add_argument("--input_height", type=int , default = 130  )
parser.add_argument("--input_width", type=int , default = 384 )
parser.add_argument("--train_batch_size", type = int, default = 32 )
parser.add_argument("--image_normalization", type = str , default = "sub_mean")

args = parser.parse_args()

import tensorflow as tf
# tf.enable_resource_variables()

# num_calibration_steps = 4

# train_generator  = LoadBatches.imageSegmentationGenerator( args.train_images , 
#                                              args.train_annotations,
#                                              1,  args.n_classes , args.input_height , args.input_width , 
#                                              16 , 48, imgNorm=args.image_normalization,)

# def representative_dataset_gen():
#     counter = 0
#     for batch in train_generator:
#     # Get sample input data as a numpy array in a method of your choosing.
#         if counter > num_calibration_steps:
#             break
#         counter += 1
#         print(batch[0][0].shape)
#         yield [batch[0][0]]


converter = tf.lite.TFLiteConverter.from_keras_model( args.model) # ,custom_objects={'iou_score':metrics_and_losses.iou_score})
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.representative_dataset = representative_dataset_gen

#converter.post_training_quantize=False

# converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]


tfmodel = converter.convert()
open ("quantization_test/converted_and_quantized.tflite" , "wb") .write(tfmodel)
