#! /bin/bash
#SBATCH --output=training_console/cityscapes_students_nokd.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/



# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes plain sm_b" \
# --experiment_name="nokd cityscapes students " --dataset="cityscapes" --options='{"preset":"sm_b"}' \
# --batch_size=32 --epochs=450

# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes plain tiny_b" \
# --experiment_name="nokd cityscapes students " --dataset="cityscapes" --options='{"preset":"tiny_b"}' \
# --batch_size=32 --epochs=450

# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes plain lg_b" \
# --experiment_name="nokd cityscapes students " --dataset="cityscapes" --options='{"preset":"lg_b"}' \
# --batch_size=32 --epochs=450

###mobilenets
# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="cityscapes plain mob0.2" \
# --experiment_name="nokd cityscapes students " --dataset="cityscapes" --options='{"alpha":0.2, "weights":null}' \
# --batch_size=32 --epochs=600

# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="cityscapes plain mob0.3" \
# --experiment_name="nokd cityscapes students " --dataset="cityscapes" --options='{"alpha":0.3, "weights":null}' \
# --batch_size=32 --epochs=600

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="cityscapes plain mob1.0" \
--experiment_name="nokd cityscapes students " --dataset="cityscapes" --options='{"alpha":1.0, "weights":null}' \
--batch_size=32 --epochs=600

# srun python3 ntrain.py --model_name="segnet_dropout" --run_name="cityscapes plain segnet" \
# --experiment_name="nokd cityscapes students " --dataset="cityscapes"  \
# --batch_size=32 --epochs=450