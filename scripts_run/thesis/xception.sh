#! /bin/bash
#SBATCH --output=training_console/xception_debug.txt
module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

srun python3 ntrain.py --experiment_name="teacher plain" --run_name="deeplab plain" \
--model_name="xception_deeplab" --batch_size=8 --epochs=300