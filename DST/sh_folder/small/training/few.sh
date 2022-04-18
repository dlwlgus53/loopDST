#!/bin/sh
#SBATCH -J PPTODfew # Job name
#SBATCH -o  ./out/t5woz.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A5000    # queue  name  or  partiton name

#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:4
#SBTACH   --ntasks=4
#SBATCH   --tasks-per-node=4
#SBATCH     --mail-user=jihyunlee@postech.ac.kr
#SBATCH     --mail-type=ALL

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module purge
module add cuda/10.4
module add cuDNN/cuda/10.4/8.0.4.30
#module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA_new "
conda activate QA_new

export PYTHONPATH=.

python ../../../learn.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --model_name t5-small\
    --ckpt_save_path ../../../ckpt/small/few_training/\
    --epoch_num 10\
    --gradient_accumulation_steps 4\
    --number_of_gpu 4\
    --batch_size_per_gpu 16\
    --train_data_ratio 0.1