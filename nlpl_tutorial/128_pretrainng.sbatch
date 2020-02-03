#!/bin/bash

# Definining resource we want to allocate. We set 8 tasks, 4 tasks over 2 nodes as we have 4 GPUs per node.
#SBATCH --nodes=2
#SBATCH --ntasks=8

# 6 CPU cores per task to keep the parallel data feeding going. A little overkill, but CPU time is very cheap compared to GPU time.
#SBATCH --cpus-per-task=6

# Enough memory.
#SBATCH --mem=64G
#SBATCH -p gpu

# Time limit on Puhti's gpu partition is 3 days.
#SBATCH -t 72:00:00
#SBATCH -J 128_pretraining

# Log file locations, %j corresponds to slurm job id.
#SBATCH -o /path/to/log_file_128_out-%j.txt
#SBATCH -e /path/to/log_file_128_err-%j.txt

# Allocate 4 GPUs on each node.
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4

# Puhti project number, you'll have to change this.
#SBATCH --account=project_name
#SBATCH

# This is used to make checkpoints and logs to readable and writable by other members in the project.
umask 0007

# Clear all modules and load Tensorflow with Horovod support.
module purge
module load tensorflow/1.13.1-hvd

# Some handy variables, you'll need to change these.
export BERT_DIR=/path/to/git_clone/DeepLearningExamples/TensorFlow/LanguageModeling/BERT_nonscaling/
export OUTPUT_DIR=/path/to/output/

mkdir -p $OUTPUT_DIR

cd $BERT_DIR

export NCCL_DEBUG=INFO

# The actual command we want to run.
# Batch size is the max amount we can fit into VRAM, `--max_seq_length` is 128 for the first part of the training.
# `--max_predictions_per_seq` is the default and must be the same as set in the tfrecord generation.
# Lastly, `--horovod` enables Horovod support, `--xla` enables TF's XLA JIT and `--use_fp16` enables support for mixed-precision training.

srun python run_pretraining.py --input_file=/path/to/input/tfrecords/cased/128/* --output_dir=$OUTPUT_DIR --do_train=True --do_eval=False --bert_config_file=/path/to/my_bert_config.json --train_batch_size=140 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=900000 --num_warmup_steps=9000 --learning_rate=1e-4 --horovod --use_xla --use_fp16

# Prints some info (CPU time used, peak RAM..) about the job at the end. Note that if the job times out this doesn't run.
seff $SLURM_JOBID
