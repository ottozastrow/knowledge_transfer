#! /bin/bash
#SBATCH --output=training_console/pretraining.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# srun python3 ntrain.py --model_name="shufflenet_hr" --options='{"preset":"sm_b"}' \
# --epochs=1000 \
# --run_name="shufflenet plain long" --experiment_name="long vs kd"

srun python3 ntrain.py --model_name="mobilenetv2_unet" --options='{"alpha":0.35, "weights":null}' \
--epochs=450 \
--run_name="mob0.35 no pretraining" --experiment_name="lr compare_pretrained_imagenet"

srun python3 ntrain.py --model_name="mobilenetv2_unet" --options='{"alpha":0.35, "weights":"imagenet"}' \
--epochs=450 \
--run_name="mob0.35 imagenet" --experiment_name="lr compare_pretrained_imagenet"



