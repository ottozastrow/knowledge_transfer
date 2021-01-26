#! /bin/bash
#SBATCH --output=training_console/subclassesdebug.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# python3 ntrain.py --subclasses_per_class=2 \
# --batch_size=48 --subclass_beta=0.001 --visualize --model_name="mobilenetv2_unet" \
# --epochs=600 --run_name="mob subclass" --experiment_name="subclass" --options='{"alpha":1.0, "weights":null}'


# python3 ntrain.py --visualize --subclasses_per_class=3 \
# --batch_size=32 --subclass_beta=0.001 --visualize \
# --initialize_weights="checkpoint_weights/subclass/run_deeplab_subclass_2020-04-27 08:35:03.714465/model-200-0.8879-val_miou_score.h5" \
# --epochs=1 --evaluate --model_name="xception_deeplab"

python3 ntrain.py --visualize --subclasses_per_class=3 \
--batch_size=32 --subclass_beta=0.001 --visualize \
--teacher="checkpoint_weights/subclass/run_deeplab_subclass_2020-04-27 08:35:03.714465/model-200-0.8879-val_miou_score.h5" \
--epochs=600 --model_name="shufflenet_hr" --model_options='{"preset":"sm_b"}' \
--distillation="hinton_distillation" --teacher_name="xception_deeplab" --experiment_name="subclass kd"

#segnet subclass plain:
# checkpoint_weights/debug/run__2020-04-27 00:30:06.838643/model-275-0.8975-val_miou_score.h5