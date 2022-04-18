#!/bin/sh
#SBATCH -J PPTODfew_I # Job name
#SBATCH -o  ./out/t5woz.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A5000    # queue  name  or  partiton name

#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:1
#SBTACH   --ntasks=1
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



CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --model_name t5-small\
    --pretrained_path ../../../ckpt/small/few_training/\
    --output_save_path ../../../inference_result/small/few_training/\
    --number_of_gpu 1\
    --batch_size_per_gpu 8