#! /bin/bash
#SBATCH --output=training_console/v5mob.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="v5 noKD mob 0.3" \
--experiment_name="v5 finetuning mob nokd" \
--batch_size=96 --epochs=450 --options='{"weights":null, "alpha":0.3}' --learning_rate=0.0005 

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="v5 noKD mob 0.25" \
--experiment_name="v5 finetuning mob nokd" \
--batch_size=96 --epochs=450 --options='{"weights":null, "alpha":0.25}' --learning_rate=0.0005 

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="v5 noKD mob 0.2" \
--experiment_name="v5 finetuning mob nokd" \
--batch_size=96 --epochs=450 --options='{"weights":null, "alpha":0.2}' --learning_rate=0.0005 

########### KD

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="v5 mob1 -> mob0.3" \
--experiment_name="v5 finetuning mob KD" \
--batch_size=96 --epochs=450 --options='{"weights":null, "alpha":0.3}' --learning_rate=0.0005 \
--teacher_name="mobilenetv2_unet" --teacher_output="teacher_output/mob teacher compare/v5_mob1/"

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="v5 mob1 -> mob0.25" \
--experiment_name="v5 finetuning mob KD" \
--batch_size=96 --epochs=450 --options='{"weights":null, "alpha":0.25}' --learning_rate=0.0005 \
--teacher_name="mobilenetv2_unet" --teacher_output="teacher_output/mob teacher compare/v5_mob1/"

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="v5 mob1 -> mob0.2" \
--experiment_name="v5 finetuning mob KD" \
--batch_size=96 --epochs=450 --options='{"weights":null, "alpha":0.2}' --learning_rate=0.0005 \
--teacher_name="mobilenetv2_unet" --teacher_output="teacher_output/mob teacher compare/v5_mob1/"


