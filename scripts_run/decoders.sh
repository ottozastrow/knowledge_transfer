#! /bin/bash
#SBATCH --output=training_console/modular.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


python3 ntrain.py \
--batch_size=16 \
--epochs=450 --experiment_name="modular" --run_name="unet down4 up1" --model_name="segnet_modular" \
--options='{"up_blocks":1, "down_blocks":4, "modular_type":"unet"}'


python3 ntrain.py \
--batch_size=16 \
--epochs=450 --experiment_name="modular" --run_name="unet down5 up2" --model_name="segnet_modular" \
--options='{"up_blocks":2, "down_blocks":5, "modular_type":"unet"}'

python3 ntrain.py \
--batch_size=16 \
--epochs=450 --experiment_name="modular" --run_name="unet down4 up2" --model_name="segnet_modular" \
--options='{"up_blocks":2, "down_blocks":4, "modular_type":"unet"}'