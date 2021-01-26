#! /bin/bash
#SBATCH --output=training_console/classifyKD2.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/


# python3 ntrain.py --classification --dataset="cifar10" \
# --model_name="cifartiny" --experiment_name="cifartiny KD experiments" \
# --run_name="vggcifar ep39 -> cifartiny" --epochs=300 --batch_size=64 \
# --teacher_name="vggcifar" --distillation="hinton_distillation" \
# --teacher="checkpoint_weights/train classifiers noKD/run_vggcifar_2020-04-19 13:25:56.862569/model-39-0.8444-val_acc.h5"

python3 ntrain.py --classification --dataset="cifar10" \
--model_name="cifartiny" --experiment_name="cifartiny KD experiments" \
--run_name="vggcifar ep70 -> cifartiny" --epochs=300 --batch_size=64 \
--teacher_name="vggcifar" --distillation="hinton_distillation" \
--teacher="checkpoint_weights/train classifiers noKD/run_vggcifar_2020-04-19 13:25:56.862569/model-70-0.8510-val_acc.h5"