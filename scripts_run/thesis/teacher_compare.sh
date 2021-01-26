#! /bin/bash
#SBATCH --output=training_console/debug_teacher_compare.txt
module load anaconda/3

export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# srun python3 ntrain.py --model_name="mobilenetv2_unet" \
# --run_name="mob1 -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
# --experiment_name="teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w1_nokd_model-264-0.9296-val_miou_score.h5"

srun python3 ntrain.py --model_name="shufflenet_hr" \
--run_name="mob1 -> sm_b" --options='{"preset":"sm_b"}' \
--experiment_name="teacher compare" \
--batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
--distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w1_nokd_model-264-0.9296-val_miou_score.h5"

srun python3 ntrain.py --model_name="shufflenet_hr" \
--run_name="mob1 -> tiny_b" --options='{"preset":"tiny_b"}' \
--experiment_name="teacher compare" \
--batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
--distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w1_nokd_model-264-0.9296-val_miou_score.h5"



# srun python3.7 ntrain.py --model_name="shufflenet_hr" \
# --run_name="lg_b -> sm_b" --options='{"preset":"sm_b"}' \
# --experiment_name="teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="shufflenet_hr" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/lg_b_model-206-0.9181-val_miou_score.h5"

# srun python3.7 ntrain.py --model_name="shufflenet_hr" \
# --run_name="sm_b -> sm_b" --options='{"preset":"sm_b"}' \
# --experiment_name="teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="shufflenet_hr" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/sm_b model-272-0.9077-val_miou_score.h5"

# srun python3.7 ntrain.py --model_name="shufflenet_hr" \
# --run_name="sm_b -> tiny_b" --options='{"preset":"tiny_b"}' \
# --experiment_name="teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="shufflenet_hr" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/sm_b model-272-0.9077-val_miou_score.h5"

# srun python3.7 ntrain.py --model_name="shufflenet_hr" \
# --run_name="lg -> tiny_b" --options='{"preset":"tiny_b"}' \
# --experiment_name="teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="shufflenet_hr" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/lg_model-150-0.9183-val_miou_score.h5"

