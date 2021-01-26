#! /bin/bash
#SBATCH --output=training_console/cutmix.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# python3 ntrain.py \
# --batch_size=32 \
# --epochs=600 --experiment_name="plain" --run_name="bce loss"

python3 ntrain.py \
--batch_size=8 \
--epochs=400 --experiment_name="plain" --run_name="plain"

