#! /bin/bash
#SBATCH --output=training_console/cutmix.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


wandb agent ottozastrow/seg_practical/wj4dntq0
