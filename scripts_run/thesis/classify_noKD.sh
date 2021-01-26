#! /bin/bash
#SBATCH --output=training_console/classify.txt

module load anaconda/3
export PATH=/home/ozastrow/cuda10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ozastrow/cuda10.1/lib64/



# python3 ntrain.py --classification --dataset="cifar10" \
# --model_name="cifartiny" --experiment_name="cifar " \
# --run_name="cifartiny plain small batchsize" --epochs=800 --batch_size=64

# python3 ntrain.py --classification --dataset="cifar10" \
# --model_name="resnet56" --experiment_name="cifar augment" \
# --run_name="resnet56 augment" --epochs=300 --batch_size=256 --augment


python3 ntrain.py --classification --dataset="cifar10" \
--model_name="vggcifar" --experiment_name="train classifiers noKD" \
--run_name="vggcifar"  --epochs=400 --batch_size=128
