#! /bin/bash
#SBATCH --output=training_console/shufflenet_lawnbot_debug.txt
module load anaconda/3

export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --model_name="shufflenetv2" \
#  --train_batch_size=64 --visualize \
#  --run_name="shufflenet divide" --experiment_name="preprocessing" \
#  --image_normalization="divide"

# srun python3 ntrain.py --model_name="segnet_dropout" \
#  --train_batch_size=64  \
#  --run_name="segnet augment per image std" --experiment_name="preprocessing" \
#  --image_normalization="per_image_standardization" --augment --visualize

srun python3 ntrain.py --model_name="shufflenet_hr" --options='{"preset":"lg_b"}' \
--experiment_name="teacher plain" --run_name="lg_b plain" --epochs=450 --batch_size=32