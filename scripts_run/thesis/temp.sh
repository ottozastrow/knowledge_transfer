#! /bin/bash
#SBATCH --output=training_console/temp.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


srun python3 ntrain.py \
--run_name="plain segnet512 small bs" --experiment_name="plain segnet512" \
--model_name="segnet_dropout_512" --batch_size=16 \
--epochs=300