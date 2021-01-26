#! /bin/bash
#SBATCH --output=training_console/segnet.txt
module load anaconda/3

export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --batch_size=32 \
# --experiment_name="segnet" --run_name="deeplab -> segnet" --distillation=hinton_distillation \
# --teacher_name="xception_deeplab" --epochs=200 \
# --teacher_output="teacher_output/KD early teacher/run_deeplab -> sm_b early teacher ep246_2020-04-19 15:46:20.507374"

srun python3 ntrain.py --batch_size=32 \
--experiment_name="segnet" --run_name="segnet -> segnet" --distillation=hinton_distillation \
--teacher_name="segnet_dropout" --epochs=200 \
--teacher="checkpoint_weights/segnet/run_plain segnet_2020-04-20 11:54:55.318601/model-127-0.9190-val_miou_score.h5"
