#! /bin/bash
#SBATCH --output=training_console/attention_transfer_wihoutkd.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain lg_b_2020-04-17 12:19:26.718204/model-330-0.6356-val_miou_score.h5" \
# --teacher_name="shufflenet_hr" --AT --teacher_options='{"preset":"lg_b"}' --dataset="cityscapes" \
# --model_name="shufflenet_hr" --experiment_name="AT" --batch_size=32 --options='{"preset":"sm_b"}' --epochs=600 --run_name="AT without KD" --at_without_kd

srun python3.7 ntrain.py --distillation="hinton_distillation" \
--teacher="checkpoint_weights/shufflnet finetuning/run_shuff lg_b_2020-03-07 14:15:50.854628/model-206-0.9181-val_miou_score.h5" \
--teacher_name="shufflenet_hr" --AT --teacher_options='{"preset":"lg_b"}' \
--model_name="shufflenet_hr" --experiment_name="AT" --batch_size=32 --options='{"preset":"sm_b"}' \
--epochs=600 --run_name="AT without KD lawnbot" --at_without_kd