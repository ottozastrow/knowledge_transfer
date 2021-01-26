#! /bin/bash
#SBATCH --output=training_console/refinery.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --distillation=hinton_distillation --stages=4 --BAN \
# --run_name="3x self distill segnet" --experiment_name="self distillation" \
# --teacher_name="segnet_dropout" --epochs=300 \
# --teacher="selected_trained_models/lawnbot/res 128x256/segnet/model-289-0.9196-val_miou_score.h5"

srun python3 ntrain.py --distillation=hinton_distillation \
--stages=3 --BAN --delete_distill_dataset \
--run_name="3x300 self distill sm_b" --experiment_name="self distillation" \
--teacher_name="shufflenet_hr" --teacher_options='{"preset":"sm_b"}' \
--epochs=300 --model_name="shufflenet_hr" --options='{"preset":"sm_b"}' \
--teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-424-0.9096-val_miou_score.h5"

# srun python3 ntrain.py --distillation=hinton_distillation --stages=3 --BAN \
# --run_name="3x self distill mob cityscapes" --experiment_name="self distillation" \
# --teacher_name="mobilenetv2_unet" --teacher_options='{"weights":null, "alpha":0.2}' \
# --epochs=300 --model_name="mobilenetv2_unet" --options='{"weights":null, "alpha":0.2}' --dataset=cityscapes \
# --teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain mob0.2_2020-04-17 19:38:54.920370/model-483-0.6570-val_miou_score.h5"