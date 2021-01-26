#! /bin/bash
#SBATCH --output=training_console/plain_kd.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# --teacher="selected_trained_models/lawnbot/res 128x256/deeplab/model-246-0.9422-val_miou_score.h5" \


srun -w octane002 --gres=gpu:4 --mem=20G python3 ntrain.py --model_name="shufflenet_hr" \
--options='{"preset":"sm_b"}' \
--run_name="plainKD_sm_b" --experiment_name="v4_plainKD_various_students" \
--teacher_output="teacher_output/save_output/run_deeplab output/" \
--batch_size=16 --epochs=300 --distillation="hinton_distillation" --debug

srun python3 ntrain.py --model_name="shufflenet_hr" \
--options='{"preset":"tiny_b"}' \
--run_name="plainKD_tiny_b" --experiment_name="v4_plainKD_various_students" \
--teacher_output="teacher_output/save_output/run_deeplab output/" \
--batch_size=32 --epochs=300 --distillation="hinton_distillation"

srun python3 ntrain.py --model_name="shufflenet_hr" \
--options='{"preset":"lg_b"}' \
--run_name="plainKD_lg_b" --experiment_name="v4_plainKD_various_students" \
--teacher_output="teacher_output/save_output/run_deeplab output/" \
--batch_size=8 --epochs=300 --distillation="hinton_distillation"



srun python3 ntrain.py --model_name="mobilenetv2_unet" \
--options='{"weights":null, "alpha":0.2}' \
--run_name="plainKD_mob0.2" --experiment_name="v4_plainKD_various_students" \
--teacher_output="teacher_output/save_output/run_deeplab output/" \
--batch_size=32 --epochs=200 --distillation="hinton_distillation"

srun python3 ntrain.py --model_name="mobilenetv2_unet" \
--options='{"weights":null, "alpha":0.25}' \
--run_name="plainKD_mob0.25" --experiment_name="v4_plainKD_various_students" \
--teacher_output="teacher_output/save_output/run_deeplab output/" \
--batch_size=32 --epochs=200 --distillation="hinton_distillation"

srun python3 ntrain.py --model_name="mobilenetv2_unet" \
--options='{"weights":null, "alpha":0.3}' \
--run_name="plainKD_mob0.3" --experiment_name="v4_plainKD_various_students" \
--teacher_output="teacher_output/save_output/run_deeplab output/" \
--batch_size=32 --epochs=200 --distillation="hinton_distillation"

