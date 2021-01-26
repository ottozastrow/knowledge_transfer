#! /bin/bash
#SBATCH --output=training_console/mobnetv2_kD.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="mob 0.4 - without imageNet" \
# --experiment_name="debug" \
# --batch_size=16 --epochs=300 --options='{"weights":null, "alpha":0.4}'


srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="mob 0.3" \
--experiment_name="v6 noKD long finetuning mob" \
--batch_size=32 --epochs=600 --options='{"weights":null, "alpha":0.3}' --learning_rate=0.0005
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-287-0.9410-val_miou_score.h5"

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="mob 0.2" \
--experiment_name="v6 noKD long finetuning mob" \
--batch_size=32 --epochs=600 --options='{"weights":null, "alpha":0.2}' --learning_rate=0.0005

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="mob 0.4" \
--experiment_name="v6 noKD long finetuning mob" \
--batch_size=32 --epochs=600 --options='{"weights":null, "alpha":0.4}' --learning_rate=0.0005
