#! /bin/bash
#SBATCH --output=training_console/takd.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# srun python3 ntrain.py --distillation=hinton_distillation --TA \
# --run_name="mob1 -> mob0.4 -> mob0.2" --experiment_name="TA" --delete_distill_dataset --epochs=450 \
# --teacher_name="mobilenetv2_unet" --teacher="checkpoint_weights/v2 finetuning mob nokd/run_mob 1.0 no pretraining_2020-04-05 23:34:14.489061/model-582-0.9314-val_miou_score.h5" \
# --model_names_list "mobilenetv2_unet" "mobilenetv2_unet" \
# --model_options_list '{"alpha":0.4, "weights":null}' '{"alpha":0.2, "weights":null}' \
# --model_name="mobilenetv2_unet"


srun python3 ntrain.py --distillation=hinton_distillation --TA \
--run_name="lg_b -> sm_b -> tiny_b cityscapes" --experiment_name="TA" --delete_distill_dataset --epochs=450 \
--teacher_name="shufflenet_hr" --teacher="checkpoint_weights/nokd cityscapes students /run_cityscapes plain lg_b_2020-04-17 12:19:26.718204/model-330-0.6356-val_miou_score.h5" \
--model_names_list "shufflenet_hr" "shufflenet_hr" \
--model_options_list '{"preset":"sm_b"}' '{"preset":"tiny_b"}' \
--model_name="shufflenet_hr" --dataset=cityscapes --temp=6

# srun python3 ntrain.py --distillation=hinton_distillation \
# --run_name="continue lg_b -> sm_b -> tiny_b cityscapes" --experiment_name="TA" --epochs=450 \
# --delete_distill_dataset --dataset=cityscapes --temp=5 \
# --model_name="shufflenet_hr" --options='{"preset":"tiny_b"}' --teacher_name="shufflenet_hr" \
# --teacher="checkpoint_weights/TA/run_lg_b -> sm_b -> tiny_b cityscapes_2020-04-23 10:48:09.827084/model-418-0.6590-val_output_miou_score.h5"