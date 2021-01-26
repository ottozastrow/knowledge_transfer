#! /bin/bash
#SBATCH --output=training_console/KD_early_teacher3.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/





# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep48" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=300 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-48-0.9279-val_miou_score.h5"

# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep78" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=450 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-78-0.9326-val_miou_score.h5"

# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep88" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=450 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-88-0.9338-val_miou_score.h5"


# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep96" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=450 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-96-0.9355-val_miou_score.h5"


# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep120" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=450 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-120-0.9369-val_miou_score.h5"

# python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep168" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=450 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-168-0.9405-val_miou_score.h5"

srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep246" \
--experiment_name="KD early teacher" \
--batch_size=16 --epochs=450 --teacher_name="xception_deeplab"  \
--distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-246-0.9422-val_miou_score.h5"


####################

# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep40" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=300 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-40-0.9267-val_miou_score.h5"


# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep54" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=300 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-54-0.9297-val_miou_score.h5"

# srun python3.7 ntrain.py --model_name="shufflenet_hr" --run_name="deeplab -> sm_b early teacher ep69" \
# --experiment_name="KD early teacher" \
# --batch_size=16 --epochs=300 --teacher_name="xception_deeplab"  \
# --distillation="hinton_distillation" --teacher="checkpoint_weights/teacher plain training/run_lr deeplab plain_2020-03-06 15:42:54.659508/model-69-0.9307-val_miou_score.h5"

