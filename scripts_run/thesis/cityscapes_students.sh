#! /bin/bash
#SBATCH --output=training_console/cityscapes_students2.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/



# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes deeplab-> sm_b" \
# --experiment_name="cityscapes students" --dataset="cityscapes" --options='{"preset":"sm_b"}' \
# --batch_size=32 --epochs=450 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes deeplab-> tiny_b" \
# --experiment_name="cityscapes students" --dataset="cityscapes" --options='{"preset":"tiny_b"}' \
# --batch_size=32 --epochs=450 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

# srun python3 ntrain.py --model_name="shufflenet_hr" --run_name="cityscapes deeplab-> lg_b" \
# --experiment_name="cityscapes students" --dataset="cityscapes" --options='{"preset":"lg_b"}' \
# --batch_size=32 --epochs=450 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

###mobilenets
# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="cityscapes deeplab-> mob0.2" \
# --experiment_name="cityscapes students" --dataset="cityscapes" --options='{"alpha":0.2, "weights":null}' \
# --batch_size=32 --epochs=600 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="cityscapes deeplab-> mob0.3" \
# --experiment_name="cityscapes students" --dataset="cityscapes" --options='{"alpha":0.3, "weights":null}' \
# --batch_size=32 --epochs=600 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="cityscapes deeplab-> mob0.4" \
# --experiment_name="cityscapes students" --dataset="cityscapes" --options='{"alpha":0.4, "weights":null}' \
# --batch_size=32 --epochs=600 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

# srun python3 ntrain.py --model_name="segnet_dropout" --run_name="cityscapes deeplab-> segnet" \
# --experiment_name="cityscapes students" --dataset="cityscapes"  \
# --batch_size=32 --epochs=450 --distillation="hinton_distillation" --teacher_name="xception_deeplab" \
# --teacher_output="teacher_output/cityscapes deeplab -> sm_b/run_cityscapes deeplab-> sm_b_2020-04-13 21:37:32.960700/"

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="mob1 -> mob0.2 600" \
--experiment_name="cityscapes students" --dataset="cityscapes"  \
--batch_size=32 --epochs=600 --distillation="hinton_distillation" \
--teacher_name="mobilenetv2_unet" --options='{"alpha":0.2,"weights":null}' \
--teacher_output="teacher_output/cityscapes students/run_mob1 -> mob0.2 600_2020-04-24 17:38:42.760573/"

# --teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain mob1.0_2020-04-24 09:35:05.117720/model-458-0.7177-val_miou_score.h5"
