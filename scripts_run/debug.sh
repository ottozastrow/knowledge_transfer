#! /bin/bash
#SBATCH --output=training_console/debug.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


srun python3 ntrain.py \
--batch_size=32 \
--epochs=0 --experiment_name="visualize" --visualize --evaluate --model_name="mobilenetv2_unet" --num_visualize=10 \
--initialize_weights="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w0.25_model-219-0.9131-val_miou_score.h5"





