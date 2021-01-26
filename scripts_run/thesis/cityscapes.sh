#! /bin/bash
#SBATCH --output=training_console/cityscapes2.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/



# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes deeplab-> sm_b" \
# --experiment_name="cityscapes deeplab -> sm_b" --dataset="cityscapes" \
# --batch_size=32 --epochs=600 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher="checkpoint_weights/cityscapes/run_deeplab long_2020-04-13 11:51:26.663360/model-431-0.7282-val_miou_score.h5"

# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="sm_b submean exp" \
# --experiment_name="cityscapes" --dataset="cityscapes" --options='{"preset": "sm_b"}' \
# --batch_size=64 --epochs=500

# srun python3 ntrain.py --model_name="xception_deeplab" --run_name="deeplab long" \
# --experiment_name="cityscapes" --dataset="cityscapes" \
# --batch_size=32 --epochs=800 

# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="sm_b submean long low lr" \
# --experiment_name="cityscapes" --dataset="cityscapes" --options='{"preset": "sm_b"}' \
# --batch_size=64 --epochs=1000 --learning_rate=0.0005


# srun python3 ntrain.py --model_name="mobilenetv2_unet" \
# --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain lg_b_2020-04-17 12:19:26.718204/model-330-0.6356-val_miou_score.h5" \
# --run_name="lg_b -> mob0.2" --experiment_name="teacher compare cityscapes" --dataset=cityscapes \
# --batch_size=32 --options='{"alpha":0.2, "weights":null}' --teacher_name="shufflenet_hr" --epochs=600

srun python3 ntrain.py --model_name="shufflenet_hr" \
--distillation="hinton_distillation" \
--teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain mob1.0_2020-04-24 09:35:05.117720/model-458-0.7177-val_miou_score.h5" \
--run_name="mob1 -> sm_b" --experiment_name="teacher compare cityscapes" --dataset=cityscapes \
--batch_size=32 --options='{"preset":"sm_b"}' --teacher_name="mobilenetv2_unet" --epochs=600
