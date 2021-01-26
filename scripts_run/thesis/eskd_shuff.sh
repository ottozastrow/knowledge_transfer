#! /bin/bash
#SBATCH --output=training_console/eskd_shuff.txt


module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py \
# --run_name="plain sm_b" --experiment_name="plain student" \
# --model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"sm_b"}'  \
# --epochs=450

# srun python3 ntrain.py \
# --run_name="plain tiny_b" --experiment_name="plain student" \
# --model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
# --epochs=450

# python3 ntrain.py \
# --run_name="sm_b ep36-> tiny_b" --experiment_name="eskd shuff student" --teacher_name="shufflenet_hr" \
# --model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
# --epochs=450 --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-36-0.8860-val_miou_score.h5"

# python3 ntrain.py \
# --run_name="sm_b ep71-> tiny_b" --experiment_name="eskd shuff student" --teacher_name="shufflenet_hr" \
# --model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
# --epochs=450 --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-71-0.8953-val_miou_score.h5"

# python3 ntrain.py \
# --run_name="sm_b ep129-> tiny_b" --experiment_name="eskd shuff student" --teacher_name="shufflenet_hr" \
# --model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
# --epochs=450 --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-129-0.8989-val_miou_score.h5"

# srun python3 ntrain.py \
# --run_name="sm_b ep240-> tiny_b" --experiment_name="eskd shuff student" --teacher_name="shufflenet_hr" \
# --model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
# --epochs=450 --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-240-0.9072-val_miou_score.h5"


srun python3 ntrain.py \
--run_name="sm_b ep424-> tiny_b" --experiment_name="eskd shuff student" --teacher_name="shufflenet_hr" \
--model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
--epochs=450 --distillation="hinton_distillation" \
--teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-424-0.9096-val_miou_score.h5"

srun python3 ntrain.py \
--run_name="sm_b ep64-> tiny_b" --experiment_name="eskd shuff student" --teacher_name="shufflenet_hr" \
--model_name="shufflenet_hr" --batch_size=32 --options='{"preset":"tiny_b"}' \
--epochs=450 --distillation="hinton_distillation" \
--teacher="checkpoint_weights/plain student/run_plain sm_b_2020-04-19 17:48:19.586260/model-64-0.8905-val_miou_score.h5"

