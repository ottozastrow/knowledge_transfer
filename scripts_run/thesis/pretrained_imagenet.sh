#! /bin/bash
#SBATCH --output=training_console/pretrained_imagenet.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/



# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="KD without imagenet - mob 0.35" \
# --experiment_name="lr compare_pretrained_imagenet" --teacher_name="xception_deeplab" \
# --batch_size=32 --epochs=450 --options='{"weights":null, "alpha":0.35}' \
# --distillation="hinton_distillation" --teacher_output="teacher_output/v5/xception/"  


srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="KD with imagenet - mob 0.35" \
--experiment_name="lr compare_pretrained_imagenet" --teacher_name="xception_deeplab" \
--batch_size=32 --epochs=450 --options='{"weights":"imagenet", "alpha":0.35}' \
--distillation="hinton_distillation" --teacher_output="teacher_output/v5/xception/"  

# srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="noKD without imagenet - mob 0.35" \
# --experiment_name="lr compare_pretrained_imagenet" --teacher_name="xception_deeplab" \
# --batch_size=64 --epochs=450 --options='{"weights":null, "alpha":0.35}' \
# --distillation="" 

srun python3 ntrain.py --model_name="mobilenetv2_unet" --run_name="noKD with imagenet - mob 0.35" \
--experiment_name="lr compare_pretrained_imagenet" --teacher_name="xception_deeplab" \
--batch_size=64 --epochs=450 --options='{"weights":"imagenet", "alpha":0.35}' \
--distillation="" 
