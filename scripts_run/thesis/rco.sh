#! /bin/bash
#SBATCH --output=training_console/rco3.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/

# srun python3 ntrain.py --distillation=hinton_distillation \
# --run_name="xception -> sm RCO 3x200" --experiment_name="RCO" \
# --teacher_name="xception_deeplab" --epochs=200 --RCO --delete_distill_dataset \
# --model_name="shufflenet_hr" --options='{"preset":"sm"}' \
# --teacher_weights_list \
# "checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-27-0.9094-val_miou_score.h5" \
# "checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-56-0.9325-val_miou_score.h5" \
# "checkpoint_weights/teacher plain/run_deeplab plain_2020-04-17 23:27:27.059259/model-287-0.9410-val_miou_score.h5"

# srun python3 ntrain.py --distillation=hinton_distillation \
# --run_name="v2 mob1 -> mob0.25 RCO 3x450" --experiment_name="RCO" \
# --teacher_name="mobilenetv2_unet" --epochs=450 --RCO --delete_distill_dataset \
# --model_name="mobilenetv2_unet" --options='{"alpha":0.25, "weights":null}' --learning_rate=0.0005 \
# --teacher_weights_list \
# "checkpoint_weights/v2 finetuning mob nokd/run_mob 1.0 no pretraining_2020-04-05 23:34:14.489061/model-57-0.9060-val_miou_score.h5" \
# "checkpoint_weights/v2 finetuning mob nokd/run_mob 1.0 no pretraining_2020-04-05 23:34:14.489061/model-201-0.9240-val_miou_score.h5" \
# "checkpoint_weights/v2 finetuning mob nokd/run_mob 1.0 no pretraining_2020-04-05 23:34:14.489061/model-582-0.9314-val_miou_score.h5"

# srun python3 ntrain.py --eval --initialize_weights="checkpoint_weights/RCO/run_v2 mob1 -> mob0.25 RCO 3x150_2020-04-22 13:37:38.100287/model-430-0.9511-val_output_miou_score.h5" \
# --epochs=0 --model_name=mobilenetv2_unet

srun python3 ntrain.py --distillation=hinton_distillation \
--run_name="city mob1 -> mob0.2 RCO 3x200" --experiment_name="RCO" --dataset="cityscapes" \
--teacher_name="mobilenetv2_unet" --epochs=200 --RCO --delete_distill_dataset --batch_size=32 \
--model_name="mobilenetv2_unet" --options='{"alpha":0.2, "weights":null}' \
--teacher_weights_list \
"checkpoint_weights/nokd cityscapes students /run_cityscapes plain mob1.0_2020-04-24 09:35:05.117720/model-126-0.6999-val_miou_score.h5" \
"checkpoint_weights/nokd cityscapes students /run_cityscapes plain mob1.0_2020-04-24 09:35:05.117720/model-215-0.7107-val_miou_score.h5" \
"checkpoint_weights/nokd cityscapes students /run_cityscapes plain mob1.0_2020-04-24 09:35:05.117720/model-458-0.7177-val_miou_score.h5"