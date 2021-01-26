#! /bin/bash
#SBATCH --output=training_console/batchsize.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# python3 ntrain.py \
# --batch_size=32 \
# --epochs=600 --experiment_name="debug_mioloss" --loss_name=miou

# python3 ntrain.py \
# --batch_size=8 \
# --epochs=450 --experiment_name="augment" --run_name="augment cutmix" --augment

python3 ntrain.py \
--batch_size=8 \
--epochs=450 --experiment_name="batchsize" --run_name="batchsize 8"
python3 ntrain.py \
--batch_size=16 \
--epochs=450 --experiment_name="batchsize" --run_name="batchsize 16"
python3 ntrain.py \
--batch_size=32 \
--epochs=450 --experiment_name="batchsize" --run_name="batchsize 32"
python3 ntrain.py \
--batch_size=64 \
--epochs=450 --experiment_name="batchsize" --run_name="batchsize 64"