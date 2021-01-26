#! /bin/bash
#SBATCH --output=training_console/barc.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

python3 ntrain.py \
--batch_size=8 \
--epochs=800 --experiment_name="barcodes" --inheight=512 --input_ratio=1.0 --dataset=barcodes --model_name=segnet_modular --decoder=up_plain --encoder=MobileNetV2 --upblocks=3 --visualize
