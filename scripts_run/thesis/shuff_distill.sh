#! /bin/bash
#SBATCH --output=training_console/distill_shufflenet_hr.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --model_name="shufflenet_hr" \
# --distillation="hinton_distillation" \
# --teacher_output="teacher_output/save_output/run_deeplab output/" \
# --run_name="deeplab -> sm" --experiment_name="KD lawnbot" \
# --batch_size=32 --options='{"preset":"sm"}' --teacher_name="xception_deeplab" --epochs=450

srun python3 ntrain.py --model_name="shufflenet_hr" \
--distillation="hinton_distillation" \
--teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain lg_b_2020-04-17 12:19:26.718204/model-330-0.6356-val_miou_score.h5" \
--run_name="lg_b -> tiny_b city" --experiment_name="teacher compare cityscapes" --dataset=cityscapes \
--batch_size=32 --options='{"preset":"tiny_b"}' --teacher_name="shufflenet_hr" --epochs=450
