#! /bin/bash
#SBATCH --output=training_console/eskd.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

srun python3 ntrain.py --distillation="hinton_distillation" \
--teacher_name="xception_deeplab" --run_name="ESKD deeplab -> segnet512 ep45" --experiment_name="ESKD lawnbot" \
--model_name="segnet_dropout_512" --batch_size=16 \
--epochs=300 --zero_kdloss_epoch=45 \
--teacher="checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-122-0.9396-val_miou_score.h5"

srun python3 ntrain.py --distillation="hinton_distillation" \
--teacher_name="xception_deeplab" --run_name="ESKD deeplab -> segnet512 ep30" --experiment_name="ESKD lawnbot" \
--model_name="segnet_dropout_512" --batch_size=16 \
--epochs=300 --zero_kdloss_epoch=30 \
--teacher="checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-122-0.9396-val_miou_score.h5"

srun python3 ntrain.py --distillation="hinton_distillation" \
--teacher_name="xception_deeplab" --run_name="ESKD deeplab -> segnet512 ep20" --experiment_name="ESKD lawnbot" \
--model_name="segnet_dropout_512" --batch_size=16 \
--epochs=300 --zero_kdloss_epoch=20 \
--teacher="checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-122-0.9396-val_miou_score.h5"