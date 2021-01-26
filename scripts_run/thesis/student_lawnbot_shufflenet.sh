#! /bin/bash
#SBATCH --output=training_console/shufflenet_teaching_lawnbot_debug.txt
module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

srun python3 ntrain.py --model_name="segnet_dropout" \
--run_name="xception -> segnet - old teachoutput" \
--experiment_name="test no temp scaling kd" --test_notempscale \
--batch_size=32 --epochs=300 --teacher_name="xception_deeplab" \
--distillation="hinton_distillation" --teacher_output="teacher_output/save_output/run_deeplab output/"

srun python3.7 ntrain.py --model_name="shufflenet_hr" \
--run_name="xception -> sm_b - new teachoutput" --options='{"preset":"sm_b"}' \
--experiment_name="test no temp scaling kd" --test_notempscale \
--batch_size=32 --epochs=450 --teacher_name="xception_deeplab" \
--distillation="hinton_distillation" --teacher="selected_trained_models/lawnbot/res 128x256/deeplab/model-246-0.9422-val_miou_score.h5"

srun python3.7 ntrain.py --model_name="shufflenet_hr" --options='{"preset":"sm_b"}' \
--run_name="lg_b -> sm_b - new teachoutput" \
--experiment_name="test no temp scaling kd" --test_notempscale \
--batch_size=32 --epochs=300 --teacher_name="shufflenet_hr" \
--distillation="hinton_distillation" --teacher_output="selected_trained_models/lawnbot/res 128x256/shufflenet/lg_b_model-206-0.9181-val_miou_score.h5"
