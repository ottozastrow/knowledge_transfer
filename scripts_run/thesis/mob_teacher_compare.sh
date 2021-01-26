#! /bin/bash
#SBATCH --output=training_console/mob_teacher_compare.txt
module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# srun python3 ntrain.py --model_name="shufflenet_hr" \
# --run_name="mob1 -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
# --experiment_name="mob teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w1_nokd_model-264-0.9296-val_miou_score.h5"

# srun python3 ntrain.py --model_name="shufflenet_hr" \
# --run_name="mob0.3 -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
# --experiment_name="mob teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w0.3_model-282-0.9156-val_miou_score.h5"

# srun python3 ntrain.py --model_name="shufflenet_hr" \
# --run_name="mob0.25 -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
# --experiment_name="mob teacher compare" \
# --batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
# --distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/mobilenetv2/w0.25_model-219-0.9131-val_miou_score.h5"

srun python3.7 ntrain.py --model_name="shufflenet_hr" \
--run_name="lg -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
--experiment_name="mob student compare" \
--batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
--distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/lg_model-150-0.9183-val_miou_score.h5"

srun python3.7 ntrain.py --model_name="shufflenet_hr" \
--run_name="lg_b -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
--experiment_name="mob student compare" \
--batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
--distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/lg_b_model-206-0.9181-val_miou_score.h5"

srun python3.7 ntrain.py --model_name="shufflenet_hr" \
--run_name="sm_b -> mob0.25" --options='{"weights":null, "alpha":0.25}' \
--experiment_name="mob student compare" \
--batch_size=32 --epochs=450 --teacher_name="mobilenetv2_unet" \
--distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/shufflenet/sm_b model-272-0.9077-val_miou_score.h5"
